# Synthetic Data Generation

Synthetic data generation through diffusion models.

## Installation

` pip install -r requirements.txt `


## Usage

```bash 
sh src/scripts/train.sh
sh src/scripts/test.sh
```

Use `CUDA_VISIBLE_DEVICES` to select the GPUs you want to train on.


## Wandb

Please, setup your wandb repo changing these lines in the two training scripts:

```python
if opt.debug:
    wandb.init(project="PUT_YOUR_PROJECT_NAME_PLZ", name='debug', mode='disabled')
else:
    wandb.init(project="PUT_YOUR_PROJECT_NAME_PLZ", name=opt.run_name)
```

