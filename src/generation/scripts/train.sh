#!/bin/bash

cd ./src
export PYTHONPATH=./src:$PYTHONPATH

python train_layout_diffusion.py --config ./configs/paper/layout_diffusion.yaml
python train_ours.py --config ./configs/paper/ours.yaml