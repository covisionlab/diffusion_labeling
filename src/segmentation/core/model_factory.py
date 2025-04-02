import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import torch
import segmentation_models_pytorch as smp

def model_factory(opt):
    # model_out_channels = len(opt.data.classes)
    model_out_channels = 1
    model_name = opt.model.name
    
    if model_name == 'unet_r50':
        model = smp.Unet(
            encoder_name='resnet50',
            encoder_weights=None,
            in_channels=opt.model.input_channels,
            classes=model_out_channels,
            activation='sigmoid',
            decoder_use_batchnorm = 'True'
        )
    
    elif model_name == 'unet_r18':
        model = smp.Unet(
            encoder_name='resnet18',
            encoder_weights=None,
            in_channels=opt.model.input_channels,
            classes=model_out_channels,
            activation='sigmoid',
            decoder_use_batchnorm = 'True'
        )

    else:
        raise ValueError(f'Model {model_name} not found')

    # Load weights if needed
    if opt.model.load_weights is not None:
        model.load_state_dict(torch.load(opt.model.load_weights, map_location=opt.device))


    return model
