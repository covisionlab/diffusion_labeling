import yaml
import json
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image


id_to_color = {
    1: [0, 0, 255],  # KNOT (openwood)
    2: [255, 0, 0],  # CRACK (openwood)
    3: [0, 255, 0],  # QUARTZITY (openwood)
    4: [255, 255, 0],  # RESIN (openwood)
    5: [255, 0, 255],  # MARROW (openwood)
}


def parse_opts(config_file):
    """Parse experimet hyperparameters into variables instead of a dictionary"""
    with open(config_file) as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt = json.loads(json.dumps(opt), object_hook=lambda d: SimpleNamespace(**d))

    opt.in_channels = json.loads(opt.network_config)['in_channels']
    opt.out_channels = json.loads(opt.network_config)['out_channels']
    opt.sampling_scheduler_steps = json.loads(opt.sampling_scheduler_config)['num_train_timesteps']
    
    opt.prediction_type = json.loads(opt.train_scheduler_config)['prediction_type']
    assert opt.prediction_type == json.loads(opt.sampling_scheduler_config)['prediction_type']

    opt.run_name = config_file.split('/')[-1].split('.')[0]

    return opt


def compile_viz(img, mask=None, bboxes=None, gt_imgs=None):
    """Compile a visualization of the image and the masks for wandb"""

    img = (img * 255).numpy().astype(np.uint8)
    if gt_imgs is not None:
        gt_imgs = (gt_imgs * 255).numpy().astype(np.uint8)
    if mask is not None:
        mask = (mask * 255).numpy().astype(np.uint8)
        mask[mask > 6] = 0

    # compile the image (img, mask1, mask2, mask3, ...)
    rows = []
    for i in range(img.shape[0]):
        row = []

        # Add original RGB image [H, W, 3]
        curr_img = np.ascontiguousarray(np.transpose(img[i], (1, 2, 0)))
        if gt_imgs is not None:
            curr_gt_img = np.ascontiguousarray(np.transpose(gt_imgs[i], (1, 2, 0)))
        
        # Add bounding boxes
        if bboxes is not None:
            curr_bboxes = bboxes[i]
            for bbox in curr_bboxes:
                class_id = int(bbox[0])
                if class_id == 0 or class_id > 6:
                    continue
                tl_x, tl_y, br_x, br_y = bbox[1:]
                tl_x = int(tl_x * curr_img.shape[1] - 1)
                tl_y = int(tl_y * curr_img.shape[0] - 1)
                br_x = int(br_x * curr_img.shape[1] - 1)
                br_y = int(br_y * curr_img.shape[0] - 1)
                curr_img = cv2.rectangle(curr_img, (tl_x, tl_y), (br_x, br_y), id_to_color[class_id], 2)
                if gt_imgs is not None:
                    curr_gt_img = cv2.rectangle(curr_gt_img, (tl_x, tl_y), (br_x, br_y), id_to_color[class_id], 2)
        if gt_imgs is not None:
            row.append(curr_gt_img)
        row.append(curr_img)
        
        # Add color mask [H, W, 3]
        if mask is not None:
            curr_mask = mask[i, 0]
            color_mask = np.zeros((curr_mask.shape[0], curr_mask.shape[1], 3), dtype=np.uint8)
            for k, color in id_to_color.items():
                color_mask[curr_mask == k] = color
            row.append(color_mask)
        
        # Combine all images horizontally for this batch item
        combined_row = np.hstack(row)
        rows.append(combined_row)

    # Stack all rows vertically
    final_image = np.vstack(rows)
    final_image = Image.fromarray(final_image)

    return final_image
