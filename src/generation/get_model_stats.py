import json

import torch
import numpy as np
from thop import profile
from diffusers import UNet2DModel

from utils.utils import parse_opts
from network.layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel

opt = parse_opts('./configs/paper/ours.yaml')
model_1 = UNet2DModel.from_config(json.loads(opt.network_config))
model_1.cuda()
flops_1, params_1 = profile(model_1,
                            inputs=(
                                torch.randn(1, 10, 128, 352).cuda(), 
                                torch.randn(1,).cuda()
                            ), verbose=False)

opt = parse_opts('./configs/paper/layout_diffusion.yaml')
net_config = json.loads(opt.network_config)
net_config["image_size"] = np.array(net_config["image_size"])
model_2 = LayoutDiffusionUNetModel(**net_config)
model_2.cuda()
flops_2, params_2 = profile(model_2,
                            inputs=(
                                torch.randn(1, 6, 128, 352).cuda(), 
                                torch.randn(1,).cuda(), 
                                torch.zeros(1, 20).cuda(), 
                                torch.zeros(1, 20, 4).cuda()
                            ), verbose=False)

print(f"{'Model':<30} {'Parameters':<20} {'FLOPS (GFLOPS)':<20}")
print(f"{'Ours':<30} {params_1 / 1e6:.2f}M{'':<15} {flops_1 / 1e9:.2f}")
print(f"{'Layout Diffusion':<30} {params_2 / 1e6:.2f}M{'':<15} {flops_2 / 1e9:.2f}")