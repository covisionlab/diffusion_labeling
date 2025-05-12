# Segmentation module


>**Note:** we included in the configs, ONLY for the crack class. To generate the other metrics, you need to hack into the config files. That is, you should change the row  `retain_class: 2  # change here to run with different classes` which points to the class you want.

## Usage

` python train.py --config configs/yourconfig.yaml `


## Wandb

Please, setup your wandb repo otherwise the training is going into the original project i.e. you need to change these lines in `main.py`

```python
if opt.debug:
    wandb.init(project="PUT_YOUR_PROJECT_NAME_PLZ", name='debug', mode='disabled')
else:
    wandb.init(project="PUT_YOUR_PROJECT_NAME_PLZ", name=opt.run_name)
```

