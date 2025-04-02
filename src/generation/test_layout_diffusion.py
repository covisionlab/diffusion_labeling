import os
import json
import argparse

import torch
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler
from lightning.fabric import Fabric

from utils.utils import parse_opts
from utils.pipelines import DiffPipeLayoutTest
from dataset.openwood import get_dloader_test
from network.layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel


def main():
    ###### hyperparams #######
    ##########################
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    opt = parse_opts(parser.parse_args().config)
    
    # Create save dir
    if os.path.exists(opt.save_dir) is False:
        os.makedirs(opt.save_dir)
    if os.path.exists(os.path.join(opt.save_dir, 'images')) is False:
        os.makedirs(os.path.join(opt.save_dir, 'images'))
    if os.path.exists(os.path.join(opt.save_dir, 'masks')) is False:
        os.makedirs(os.path.join(opt.save_dir, 'masks'))
    
    # Initialize fabric
    fabric = Fabric(accelerator='cuda', strategy='ddp', devices='auto')
    fabric.seed_everything(opt.seed + fabric.global_rank)
    fabric.launch()
    
    ########## model ##########
    ###########################
    net_config = json.loads(opt.network_config)
    net_config["image_size"] = np.array(net_config["image_size"])
    unet = LayoutDiffusionUNetModel(**net_config)
    sampling_scheduler = DDPMScheduler.from_config(json.loads(opt.sampling_scheduler_config))
    sampling_scheduler.set_timesteps(num_inference_steps=1000, device=opt.device)
    unet.to(fabric.device)

    ########## data ##########
    ##########################
    _, dataloader = get_dloader_test(opt)
    dataloader = fabric.setup_dataloaders(dataloader)

    ########## load model #########
    ###############################
    if opt.pretrained_model is not None:
        checkpoint = torch.load(opt.pretrained_model)
        unet.load_state_dict(checkpoint['model'])
    
    ########## inference ##########
    ###############################
    unet = fabric.setup(unet)
    unet.eval()
    
    pipeline = DiffPipeLayoutTest(model=unet, scheduler=sampling_scheduler).to(fabric.device)
    
    print('len dataloader:', len(dataloader))
    for n, batch in enumerate(dataloader):
        imgs, masks = pipeline(opt, batch, fabric.device)
        for img_name, img, mask in zip(batch['images_name'], imgs, masks):
            img = (img * 255).type(torch.uint8).permute(1, 2, 0).numpy()
            Image.fromarray(img).save(f'{opt.save_dir}/images/{img_name}.jpg')
            mask = (mask[0] * 255).type(torch.uint8).numpy()
            Image.fromarray(mask).save(f'{opt.save_dir}/masks/{img_name}.png')
        print(f'Processed {n} images / {len(dataloader)}')


if __name__ == '__main__':
    main()
