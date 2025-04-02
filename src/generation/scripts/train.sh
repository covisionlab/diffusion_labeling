#!/bin/bash

cd ./src
export PYTHONPATH=./src:$PYTHONPATH

python train_layout_diffusion.py --config ./configs/paper/layout_diffusion_openwood_from_scratch.yaml
python train_ours.py --config ./configs/paper/analog_bits_bbox_cond_sdf+classmap_openwood_from_scratch.yaml