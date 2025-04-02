
import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import os
from tqdm import tqdm
import wandb
import torch
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.fabric import Fabric
from wandb.integration.lightning.fabric import WandbLogger

from src.segmentation.core.pytorch_ext_local import F1Measure, F1MeasureMultich, BinaryDiceLoss
from src.segmentation.core.model_factory import model_factory
from src.segmentation.core.data_factory import data_factory
from src.segmentation.core.utils import visualize_predictions, parse_opts, visualize_batch
import wandb
import argparse



def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ###### Logging #######
    ######################

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/crack.yaml')
    opt, opt_dict = parse_opts(parser.parse_args().config)

    if opt.main.debug:
        print("[ dbg ] - Wandb disabled")
        logger = WandbLogger(project=opt.log.wandb_project, name='debug', mode='disabled', config=opt_dict)
    else:
        print(f"[ ok ] - Wandb enabled - run_name: {opt.run_name}")
        logger = WandbLogger(project=opt.log.wandb_project, name=opt.run_name, config=opt_dict)

    # Initialize fabric
    if opt.main.fabric_distributed:
        fabric = Fabric(accelerator=opt.main.fabric_accelerator, strategy='ddp', devices='auto', loggers=logger)
    else:
        fabric = Fabric(accelerator=opt.main.fabric_accelerator, devices='auto', loggers=logger)

    fabric.seed_everything(opt.main.seed + fabric.global_rank)
    fabric.launch()
    opt.device = fabric.device

    ######  Setup  #######
    ######################
    if opt.hyp.use_trigger:
        def trigger(ce_loss, alpha, beta):
            if ce_loss < 0.2:
                alpha = 0.3
                beta = 0.7
            return alpha, beta
    else:
        trigger = lambda x, a, b: (a, b)

    model = model_factory(opt)
    model = model.to(fabric.device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.hyp.lr, weight_decay=opt.hyp.wd)
    if opt.hyp.scheduler.type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max= opt.hyp.scheduler.T_max, eta_min= opt.hyp.scheduler.eta_min )
    
    loss_fn = nn.BCELoss()
    dice_fn = BinaryDiceLoss()

    # data
    train_dset, val_dset, train_loader, val_loader = data_factory(opt)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    model, optimizer = fabric.setup(model, optimizer)

    fabric.log_dict({"train_files": wandb.Table(dataframe=train_dset.data_df)})
    fabric.log_dict({"val_files": wandb.Table(dataframe=val_dset.data_df)})

    ######  Train  #######
    ######################

    max_f1 = 0
    alpha, beta = opt.hyp.alpha, opt.hyp.beta
    for epoch in range(opt.hyp.epochs):
        # if epoch == 1:
        #    assert 1 == 0  # debug
        model.train()
        samples_seen, sum_ce, sum_dice, sum_total, mu_f1 = 0, 0, 0, 0, 0
        measure = F1Measure()
        multich_measure = F1MeasureMultich([opt.data.classes[opt.data.retain_class-1] + "_tr"])

        for batch_n, batch in enumerate(pbar := tqdm(train_loader)):
            batch_n += 1

            x, mask = batch

            x = x.to(fabric.device)
            mask = mask.to(fabric.device).contiguous()

            # visualize tensor
            # visualize_batch(x, mask, train_dset.unnorm, f'/home/fpelosin/projects/microtec/microtec_synthetic_data_generation/layout/{batch_n}.png')

            # ford
            out = model(x)

            ce_loss = loss_fn(out, mask)
            dice_loss = dice_fn(out, mask)
            alpha, beta = trigger(ce_loss, alpha, beta)
            loss = alpha*ce_loss + beta*dice_loss


            # backward
            fabric.backward(loss)
            optimizer.step()

            # metrics
            sum_ce += ce_loss.item()
            sum_dice += dice_loss.item()
            sum_total += loss.item()
            mu_f1 = measure.update(out>opt.hyp.threshold, mask)
            much = multich_measure.update(out>opt.hyp.threshold, mask)

            # log
            pbar.set_description(f"[EPOCH {epoch:03} train] ce:{sum_ce/batch_n:.4f} dice:{sum_dice/batch_n:.4f} tot:{sum_total/batch_n:.4f} m_bf1:{mu_f1:.4f}")
            if not opt.main.debug:
                fabric.log_dict({'CE/train': sum_ce/batch_n, 'Dice/train': sum_dice/batch_n, 'F1/train': mu_f1})
                fabric.log_dict(much)

            optimizer.zero_grad()
        scheduler.step()

        #######  Val  ########
        ######################

        if (epoch % opt.log.log_every == 0) and epoch > 0:

            print(opt.run_name , f' - current best F1: {max_f1:.3f}')
            
            with torch.inference_mode(True):
                model.eval()
                samples_seen, sum_ce, sum_dice, sum_total, mu_f1 = 0, 0, 0, 0, 0
                measure = F1Measure()
                multich_measure = F1MeasureMultich([opt.data.classes[opt.data.retain_class-1] + "_val"])
                
                # bnorm = 1 always
                for batch in (pbar := tqdm(val_loader)):
                    
                    x, mask = batch

                    
                    x = x.to(fabric.device)
                    mask = mask.to(fabric.device)

                    out = model(x)

                    ce_loss = loss_fn(out, mask)
                    dice_loss = dice_fn(torch.sigmoid(out), mask)
                    loss = alpha*ce_loss + beta*dice_loss

                    # metrics
                    sum_ce += ce_loss.item()
                    sum_dice += dice_loss.item()
                    sum_total += loss.item()
                    mu_f1 = measure.update(out>opt.hyp.threshold, mask)
                    much = multich_measure.update(out>opt.hyp.threshold, mask)
                    samples_seen += x.shape[0]

                    # log   
                    pbar.set_description(f"[EPOCH {epoch:03} val] ce:{sum_ce/samples_seen:.4f} dice:{sum_dice/samples_seen:.4f} tot:{sum_total/samples_seen:.4f} m_bf1:{mu_f1:.4f}")

                if not opt.main.debug:
                    fabric.log_dict({'CE/val': sum_ce/samples_seen,'Dice/val': sum_dice/samples_seen,'F1/val': mu_f1})
                    fabric.log_dict(much)

                    # visualize predictions in wandb
                    visualize_predictions(model, val_loader, val_dset.unnorm, fabric, threshold=opt.hyp.threshold, num_samples=opt.log.show_n)

                if mu_f1 > max_f1:
                    max_f1 = mu_f1
                    print(f'[ !!! ] -.-.-.-.- New best f1: {mu_f1:.3f} -.-.-.-.- [ !!! ]')
                    os.makedirs(opt.log.checkpoint_folder + f'/{opt.run_name}', exist_ok=True)
                    fname = opt.log.checkpoint_folder + f'/{opt.run_name}/best.pth'
                    torch.save(model.state_dict(), fname)


if __name__ == '__main__':
    print("Starting training")
    main()
