# Synthetic Data Generation

Synthetic data generation through diffusion models.


1. `preprocessing/`: in this folder you create our bbox encodings. Used to train our method.
2. `scripts/train.sh`: with this you train the diffusion pipelines both ours and layout-diffusion.
3. `scripts/test.sh`: with this you generate the synthetic wood boards.

## Usage

1. run `preprocessing/compute_bbox_maps.py` by altering the input files.

2. First, train the diffusion pipeline with:

```bash 
sh scripts/train.sh
```

3. Secondly, generate the synthetic examples with

```bash
sh scripts/test.sh
```

Make sure you set up the output folder for the `test.sh`. Check in the config files for `save_dir: "./output/ours"` to choose the output folders.

Use `CUDA_VISIBLE_DEVICES` to select the GPUs you want to train on.

## Wandb

Please, setup your wandb repo changing these lines in the two training scripts:

```python
if opt.debug:
    wandb.init(project="PUT_YOUR_PROJECT_NAME_PLZ", name='debug', mode='disabled')
else:
    wandb.init(project="PUT_YOUR_PROJECT_NAME_PLZ", name=opt.run_name)
```

