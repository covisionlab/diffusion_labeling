import yaml
from types import SimpleNamespace
import torch
import numpy as np
import wandb
import cv2

def visualize_infer_batch(x, mask_hat, mask, unnorm, name):
    
    # x is shape b x c x h x w
    x = x.cpu()
    x = unnorm(x)

    # mask_hat is shape b x 1 x  h x w
    mask_hat = mask_hat.cpu()

    # mask is shape b x 1 x  h x w
    mask = mask.cpu()

    stacked_tensor = []
    # visualize img, mask, and mask_hat vertically
    for i in range(x.shape[0]):
        # stack vertically the image and the masks
        img = x[i].permute(1, 2, 0).numpy()

        mask_img = mask[i].permute(1, 2, 0).numpy()
        mask_img = mask_img.repeat(3, axis=-1)
        
        mask_hat_img = mask_hat[i].permute(1, 2, 0).numpy()
        mask_hat_img = mask_hat_img.repeat(3, axis=-1)
        
        # Add green lines to separate the images
        width = img.shape[1]
        separator = np.zeros((5, width, 3), dtype=np.uint8)
        separator[:, :, 1] = 1  # Green line
        
        # Stack vertically with separators
        stacked = np.vstack([img, separator, mask_img, separator, mask_hat_img])
        stacked_tensor.append(stacked)

    stacked_tensor = np.vstack(stacked_tensor)
    assert cv2.imwrite(name, (stacked_tensor * 255).astype(np.uint8))


def visualize_batch(x, mask, unnorm, name):
    
    # x is shape b x c x h x w
    x = x.cpu()
    x = unnorm(x)

    # mask is shape b x 1 x  h x w
    mask = mask.cpu()

    stacked_tensor = []
    # visualize img,mask horizontally
    for i in range(x.shape[0]):
        # stack horizontally the image and the mask
        img = x[i].permute(1,2,0).numpy()

        mask_img = mask[i].permute(1,2,0).numpy()
        mask_img = mask_img.repeat(3, axis=-1)
        # stack horizontally
        stacked = np.hstack([img, mask_img])
        stacked_tensor.append(stacked)

    stacked_tensor = np.vstack(stacked_tensor)
    assert cv2.imwrite(name, (stacked_tensor*255).astype(np.uint8))


def dict_to_namespace(d):
    """Recursively convert a dictionary to a SimpleNamespace"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def parse_opts(config_file):
    """Parse experiment hyperparameters into variables instead of a dictionary"""
    with open(config_file) as f:
        opt_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    opt = dict_to_namespace(opt_dict)  # Recursively convert
    opt.run_name = config_file.split('/')[-1].split('.')[0]

    return opt, opt_dict


def visualize_predictions(model, val_loader, unnorm, fabric, threshold=0.5, num_samples=5):
    """
    Creates side-by-side visualizations of (Input | Ground Truth | Prediction) 
    and logs them to wandb.

    Args:
    - model: Trained PyTorch model.
    - val_loader: DataLoader for validation data.
    - fabric: Lightning Fabric for device management.
    - threshold: Threshold for binarizing model output.
    - num_samples: Number of samples to visualize.
    """
    model.eval()
    
    with torch.no_grad():
        stacked_vertically = []
        for i, (x, mask) in enumerate(val_loader):
            if i >= num_samples:
                break

            x = x.to(fabric.device)
            mask = mask.to(fabric.device)

            out = model(x)
            pred_mask = (out > threshold).float()

            x = unnorm(x) * 255

            # Convert tensors to NumPy
            input_img = x[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)  # (C, H, W) → (H, W, C)
            mask_rgb = get_colored_masks(mask)
            pred_mask_rgb = get_colored_masks(pred_mask)

            stacked_img = np.hstack([input_img, mask_rgb[0].transpose(1, 2, 0), pred_mask_rgb[0].transpose(1, 2, 0)])            
            
            stacked_vertically.append(stacked_img)

        # Stack vertically
        stacked_vertically = np.vstack(stacked_vertically)

        # log to wandb
        fabric.log("visualization", wandb.Image(stacked_vertically))


def get_colored_masks(mask_in: torch.Tensor) -> np.ndarray:
    """
    Args:
        mask_in: 4D tensor with shape (B, C, H, W)
    Returns:
        colored_masks: 4D tensor with shape (B, 3, H, W)
    """
    assert mask_in.dim() == 4, 'mask_in must be 4D tensor with shape (B, C, H, W)'
    mask = mask_in.clone().detach().cpu().numpy()
    defects = [#('foreground', (0, 255, 0)),
               ('knot', (255, 145, 0)), 
               ('crack',(255, 0, 255)),
               ('quarzity', (0,0,255)), 
               ('resin',(153, 99, 0)), 
               ('marrow', (255,255,0)), ]

    # draw_order = [4,3,2,0,1]
    draw_order = [0]

    colored_masks = []
    
    # for each image
    for b in range(len(mask)):

        # create a blank image
        colored_mask = np.zeros((mask.shape[2], mask.shape[3], 3), dtype=np.uint8)
        
        # for each channel in the output output tensor
        for c in draw_order:
            roi = mask[b,c,:,:] == 1
            colored_mask[roi] = defects[c][1]
        
        colored_masks.append(colored_mask.transpose(2,0,1))
    colored_masks = np.stack(colored_masks, axis=0)
    return colored_masks


