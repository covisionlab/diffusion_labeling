#!/bin/bash

cd ./src
export PYTHONPATH=./src:$PYTHONPATH

python test_layout_diffusion.py --config ./configs/paper/test/layout_diffusion.yaml
python test_ours.py --config ./configs/paper/test/ours.yaml