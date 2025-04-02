#!/bin/bash

cd ./src
export PYTHONPATH=./src:$PYTHONPATH

python test_layout_diffusion.py --config ./configs/paper/test/layout_diffusion_openwood_from_scratch.yaml
python test_ours.py --config ./configs/paper/test/analog_bits_bbox_cond_sdf+classmap_openwood_from_scratch.yaml