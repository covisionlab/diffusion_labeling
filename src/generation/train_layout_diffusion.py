import os
import json
import argparse

import wandb
import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from lightning.fabric import Fabric
from wandb.integration.lightning.fabric import WandbLogger

from utils.pipelines import DiffPipeLayout
from utils.analog_bits import decimal_to_bits
from utils.utils import compile_viz, parse_opts
from dataset.openwood import get_dloader
from network.model_factory import get_model_scheduler_layout


def main():
    ###### hyperparams #######
    ##########################
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    opt = parse_opts(parser.parse_args().config)
    if opt.debug:
        logger = WandbLogger(project='synthetic_data_generation', name='debug', mode='disabled')
    else:
        logger = WandbLogger(project='synthetic_data_generation', name=opt.run_name, config=opt)
    
    # Create save dir
    if os.path.exists(opt.save_dir) is False:
        os.makedirs(opt.save_dir)
    
    # Initialize fabric
    fabric = Fabric(accelerator='cuda', strategy='ddp', devices='auto', loggers=logger)
    fabric.seed_everything(opt.seed + fabric.global_rank)
    fabric.launch()
    
    ########## model ##########
    ###########################
    unet, scheduler, sampling_scheduler = get_model_scheduler_layout(opt)
    unet.to(fabric.device)

    ########## data ##########
    ##########################
    dataset, train_dataloader = get_dloader(opt)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    ########## train ##########
    ###########################
    start_epoch = 0
    layout_length = json.loads(opt.network_config)['layout_encoder']['layout_length']
    num_classes_for_layout_object = json.loads(opt.network_config)['layout_encoder']['num_classes_for_layout_object']
    optimizer = torch.optim.AdamW(unet.parameters(), lr=opt.lr)
    if opt.pretrained_model is not None:
        checkpoint = torch.load(opt.pretrained_model)
        unet.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    
    unet, optimizer = fabric.setup(unet, optimizer)
    
    for epoch in range(start_epoch, opt.num_epochs):
        print(f'[ EPOCH  {epoch} ]')
        for n, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # get batch
            input_imgs = batch['images'].to(fabric.device)
            mask = batch['mask'].to(fabric.device)
            bs = input_imgs.shape[0]
            
            # Convert masks to analog bits
            mask = decimal_to_bits(mask, bits=int(np.ceil(np.log2(opt.num_classes))))
            
            # get bbox and class for layout encoder
            obj_bbox = torch.zeros([bs, layout_length, 4])
            obj_bbox[:, 0] = torch.tensor([[0, 0, 1, 1]]).repeat(bs, 1)
            obj_class = torch.LongTensor(bs, layout_length).fill_(num_classes_for_layout_object - 1)
            obj_class[:, 0] = torch.LongTensor(bs).fill_(0)
            for n, bbox in enumerate(batch['bbox']):
                curr_classes = bbox[:, 0]
                curr_bboxes = bbox[:, 1:]
                obj_bbox[n, 1:1+len(curr_bboxes)] = curr_bboxes
                obj_class[n, 1:1+len(curr_classes)] = curr_classes
            obj_bbox = obj_bbox.to(fabric.device)
            obj_class = obj_class.to(fabric.device)

            # Concatenate the masks (analog bits) to the input images
            input_imgs = torch.cat((input_imgs, mask), dim=1)

            # Sample noise to be added to the images
            noise = torch.randn(input_imgs.shape).to(input_imgs.device)

            # Sample a random timestep for each image
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=input_imgs.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            input_imgs = scheduler.add_noise(input_imgs, noise, timesteps)

            pred = unet(input_imgs, timesteps, obj_class, obj_bbox)[0]

            # Get the model prediction for the noise
            if opt.prediction_type == 'epsilon':
                loss = F.mse_loss(pred, noise)
            else:
                loss = F.mse_loss(pred, input_imgs)

            # Log the loss
            fabric.log_dict({
                'loss':loss.item()
            })

            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        ########## logging ##########
        #############################
        if (epoch != 0) and ((epoch % opt.log_samples_every) == 0):
            # Generate images and masks
            unet.eval()
            pipeline = DiffPipeLayout(model=unet, scheduler=sampling_scheduler).to(fabric.device)
            imgs, masks, bboxes, gt_imgs = pipeline(opt, dataset, fabric.device)
            final_image = compile_viz(imgs, masks, bboxes, gt_imgs)
            unet.train()
            fabric.log_dict({
                "visualizations": wandb.Image(final_image, caption="Image and Masks Visualization")
            })
        
        ########## checkpoint ##########
        ################################
        if (epoch != 0) and ((epoch % opt.save_model_every) == 0) and (fabric.global_rank == 0):
            checkpoint = {
                'epoch': epoch,
                'model': unet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
            }
            torch.save(checkpoint, f'{opt.save_dir}/{logger.experiment.name}_unet_epoch_{epoch}.pt')

    # save model at the end of all
    if fabric.global_rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model': unet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, f'{opt.save_dir}/{logger.experiment.name}_unet_epoch_{epoch}.pt')


if __name__ == '__main__':
    main()