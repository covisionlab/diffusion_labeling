import os

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def compute_sdf_classmap(shape, bboxes):
    height, width = shape
    sdf_map = np.full((height, width), np.nan, dtype=np.float32)
    class_map = np.zeros((height, width), dtype=np.float32)
    
    for class_id, x1, y1, x2, y2 in bboxes:
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * height), int(y2 * height)
        
        # Create border coordinates using NumPy
        x_range = np.arange(x1, x2 + 1)
        y_range = np.arange(y1, y2 + 1)
        border_coords = np.vstack([
            np.column_stack([x_range, np.full_like(x_range, y1)]),
            np.column_stack([x_range, np.full_like(x_range, y2)]),
            np.column_stack([np.full_like(y_range, x1), y_range]),
            np.column_stack([np.full_like(y_range, x2), y_range])
        ])
        
        # Compute distance transform efficiently
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        dx = grid_x[..., None] - border_coords[:, 0]
        dy = grid_y[..., None] - border_coords[:, 1]
        distance_to_edge = np.min(np.sqrt(dx**2 + dy**2), axis=-1)
        
        # Identify inside/outside points
        inside = (x1 < grid_x) & (grid_x < x2) & (y1 < grid_y) & (grid_y < y2)
        signed_distance = np.where(inside, distance_to_edge, -distance_to_edge)
        
        bbox_x_coords, bbox_y_coords = np.where((x1 <= grid_x) & (grid_x <= x2) & (y1 <= grid_y) & (grid_y <= y2))
        out_bbox_coords_x, out_bbox_coords_y = np.where((x1 > grid_x) | (grid_x > x2) | (y1 > grid_y) | (grid_y > y2))
        
        # Update maps where needed
        if np.isnan(sdf_map).all():
            sdf_map = signed_distance
            class_map[bbox_x_coords, bbox_y_coords] = class_id
        else:
            neg_indexes = np.where(sdf_map[bbox_x_coords, bbox_y_coords] < 0)
            if len(neg_indexes[0]) > 0:
                sdf_map[bbox_x_coords[neg_indexes], bbox_y_coords[neg_indexes]] = signed_distance[bbox_x_coords[neg_indexes], bbox_y_coords[neg_indexes]]
                class_map[bbox_x_coords[neg_indexes], bbox_y_coords[neg_indexes]] = class_id
            lower_indexes = np.where(signed_distance[bbox_x_coords, bbox_y_coords] < sdf_map[bbox_x_coords, bbox_y_coords])
            if len(lower_indexes[0]) > 0:
                sdf_map[bbox_x_coords[lower_indexes], bbox_y_coords[lower_indexes]] = signed_distance[bbox_x_coords[lower_indexes], bbox_y_coords[lower_indexes]]
                class_map[bbox_x_coords[lower_indexes], bbox_y_coords[lower_indexes]] = class_id
            higher_indexes = np.where(signed_distance[out_bbox_coords_x, out_bbox_coords_y] > sdf_map[out_bbox_coords_x, out_bbox_coords_y])
            if len(higher_indexes[0]) > 0:
                sdf_map[out_bbox_coords_x[higher_indexes], out_bbox_coords_y[higher_indexes]] = signed_distance[out_bbox_coords_x[higher_indexes], out_bbox_coords_y[higher_indexes]]
    
    diag = np.sqrt(height**2 + width**2)
    sdf_map = torch.from_numpy(sdf_map / diag)[None, ...].float()
    class_map = torch.from_numpy(class_map / 255)[None, ...]
    
    return sdf_map, class_map


class OpenWoodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, bbox_dir, split_file):
        """
        Args:
            image_dir (str): Path to the directory with images.
            mask_dir (str): Path to the directory with segmentation masks.
            bbox_dir (str): Path to the directory with bounding box annotations (in YOLO format).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bbox_dir = bbox_dir
        
        self.split_file = pd.read_csv(split_file)
        self.image_files = sorted(self.split_file['image'].tolist())
        self.image_files = [img.split('/')[-1] for img in self.image_files]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get the image filename and corresponding mask and bbox files
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.jpg', '.png')  # Assuming same name for mask
        bbox_name = img_name.replace('.jpg', '.txt')  # Assuming same name for bbox
        
        # Load the segmentation mask
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.array(Image.open(mask_path))
        
        # Load the bounding boxes
        bbox_path = os.path.join(self.bbox_dir, bbox_name)
        with open(bbox_path, 'r') as f:
            bboxes = f.readlines()
        bboxes = [list(map(float, line.strip().split())) for line in bboxes]
        bboxes = [bbox for bbox in bboxes if bbox[0] <= 5]
        
        # create sdf map and class map from bboxes
        if len(bboxes) == 0:
            sdf_map = np.full((mask.shape[1], mask.shape[2]), -1).astype(np.float32)
            class_map = np.zeros((mask.shape[1], mask.shape[2])).astype(np.float32)
            sdf_map = torch.from_numpy(sdf_map)[None, ...]
            class_map = torch.from_numpy(class_map)[None, ...]
        else:
            sdf_map, class_map = compute_sdf_classmap(mask.shape, bboxes)
        
        return {
            'images_name': img_name.split('.')[0], 
            'sdf_map': sdf_map, 
            'class_map': class_map
        }
    

if __name__ == '__main__':
    image_dir = './data/real/images'
    mask_dir = './data/real/masks'
    bbox_dir = './data/real/bbox'
    split_file = './data/splits/real_70.csv'
    # split_file = './data/splits/real_20.csv'
    
    sdf_out_dir = './data/real/sdf_map'
    class_out_dir = './data/real/class_map'
    os.makedirs(sdf_out_dir, exist_ok=True)
    os.makedirs(class_out_dir, exist_ok=True)

    dataset = OpenWoodDataset(image_dir, mask_dir, bbox_dir, split_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    for batch in tqdm(dataloader):
        name = batch['images_name'][0]
        class_map = batch['class_map'][0, 0].numpy()
        sdf_map = batch['sdf_map'][0, 0].numpy()
        np.save(sdf_out_dir + name + '.npy', sdf_map)
        np.save(class_out_dir + name + '.npy', class_map)
