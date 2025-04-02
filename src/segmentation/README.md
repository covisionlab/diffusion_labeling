# Synthetic Data Generation

Synthetic data generation through diffusion models.

## Installation

` pip install -r requirements.txt `


## Usage

` python src/main.py --config configs/hyp/yes_cond_pretr_bedroom.yaml `


## Wandb

Please, setup your wandb repo otherwise the training is going into the original project i.e. you need to change these lines in `main.py`

```python
if opt.debug:
    wandb.init(project="PUT_YOUR_PROJECT_NAME_PLZ", name='debug', mode='disabled')
else:
    wandb.init(project="PUT_YOUR_PROJECT_NAME_PLZ", name=opt.run_name)
```

