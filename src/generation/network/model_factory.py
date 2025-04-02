import json

import numpy as np
from diffusers import UNet2DModel
from diffusers import DDPMScheduler

from network.layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel


def get_model_scheduler_layout(opt):
    """Build the model and sampling_scheduler"""
    net_config = json.loads(opt.network_config)
    net_config["image_size"] = np.array(net_config["image_size"])
    unet = LayoutDiffusionUNetModel(**net_config)
    scheduler = DDPMScheduler.from_config(json.loads(opt.train_scheduler_config))

    # load the sampling scheduler
    sampling_scheduler = DDPMScheduler.from_config(json.loads(opt.sampling_scheduler_config))
    sampling_scheduler.set_timesteps(num_inference_steps=opt.sampling_scheduler_steps, device=opt.device)
      
    return unet, scheduler, sampling_scheduler


def get_model_scheduler_ours(opt):
    unet = UNet2DModel.from_config(json.loads(opt.network_config))
    scheduler = DDPMScheduler.from_config(json.loads(opt.train_scheduler_config))

    # load the sampling scheduler
    sampling_scheduler = DDPMScheduler.from_config(json.loads(opt.sampling_scheduler_config))
    sampling_scheduler.set_timesteps(num_inference_steps=opt.sampling_scheduler_steps)

    return unet, scheduler, sampling_scheduler
