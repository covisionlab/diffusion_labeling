import os

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def get_dloader(opt):
    dataset = OpenWoodDataset(
        image_dir=opt.image_dir,
        mask_dir=opt.mask_dir,
        bbox_dir=opt.bbox_dir,
        sdf_dir=opt.sdf_dir,
        class_map_dir=opt.class_map_dir,
        split_file=opt.split_file
    )
    dataset.use_subset(opt.n_subset)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    return dataset, train_dataloader


def get_dloader_test(opt):
    dataset = OpenWoodDataset(
        image_dir=opt.image_dir,
        mask_dir=opt.mask_dir,
        bbox_dir=opt.bbox_dir,
        sdf_dir=opt.sdf_dir,
        class_map_dir=opt.class_map_dir,
        split_file=opt.split_file
    )
    dataset.use_subset(opt.n_subset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    return dataset, dataloader


def custom_collate_fn(batch):
    images_name, images, masks, bboxes, sdf_maps, class_maps = [], [], [], [], [], []
    for b in range(len(batch)):
        images_name.append(batch[b][0])
        images.append(batch[b][1])
        masks.append(batch[b][2])
        bboxes.append(batch[b][3])
        sdf_maps.append(batch[b][4])
        class_maps.append(batch[b][5])
    
    # Stack images and masks along the batch dimension
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    # Keep bounding boxes as a list of tensors (no padding)
    bboxes = [torch.tensor(bbox) for bbox in bboxes]
    
    # Stack sdf_maps and class_maps along the batch dimension
    sdf_maps = torch.stack(sdf_maps, dim=0)
    class_maps = torch.stack(class_maps, dim=0)
    
    return {
        'images_name': images_name,
        'images': images,
        'mask': masks,
        'bbox': bboxes,
        'sdf_map': sdf_maps,
        'class_map': class_maps
    }


class OpenWoodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, bbox_dir, sdf_dir, class_map_dir, split_file):
        """
        Args:
            image_dir (str): Path to the directory with images.
            mask_dir (str): Path to the directory with segmentation masks.
            bbox_dir (str): Path to the directory with bounding box annotations (in YOLO format).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bbox_dir = bbox_dir
        self.sdf_dir = sdf_dir
        self.class_map_dir = class_map_dir
        
        self.split_file = pd.read_csv(split_file)
        self.image_files = sorted(self.split_file['image'].tolist())
        self.image_files = [img.split('/')[-1] for img in self.image_files]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get the image filename and corresponding mask and bbox files
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.jpg', '.png')
        bbox_name = img_name.replace('.jpg', '.txt')
        sdf_name = img_name.replace('.jpg', '.npy')
        class_map_name = img_name.replace('.jpg', '.npy')
        
        # Load the image
        img_path = os.path.join(self.image_dir, img_name)
        image = torch.from_numpy(np.array(Image.open(img_path).convert("RGB"))).permute(2, 0, 1)
        image = image / 255.
        image = (image * 2) - 1
        
        # Load the segmentation mask
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = torch.from_numpy(np.array(Image.open(mask_path))).unsqueeze(0)
        mask = mask / 255.
        
        # Load the bounding boxes
        bbox_path = os.path.join(self.bbox_dir, bbox_name)
        with open(bbox_path, 'r') as f:
            bboxes = f.readlines()
        bboxes = [list(map(float, line.strip().split())) for line in bboxes]
        bboxes = np.array(bboxes)
        
        # Load the SDF and class maps
        sdf_path = os.path.join(self.sdf_dir, sdf_name)
        sdf_map = np.load(sdf_path)
        sdf_map = torch.from_numpy(sdf_map).float()[None, ...]
        class_path = os.path.join(self.class_map_dir, class_map_name)
        class_map = np.load(class_path)
        class_map = torch.from_numpy(class_map).float()[None, ...]
        
        return img_name.split('.')[0], image, mask, bboxes, sdf_map, class_map

    def use_subset(self, n):
        """ This function is used to reduce the size of the dataset, can be called after the dataset is created"""
        self.image_files = self.image_files[:n]


if __name__ == '__main__':   
    image_dir = './data/real/images'
    mask_dir = './data/real/masks'
    bbox_dir = './data/real/bbox'
    sdf_dir = './data/real/sdf_map'
    class_map_dir = './data/real/class_map'
    split_file = './data/splits/real_20.csv'

    dataset = OpenWoodDataset(image_dir, mask_dir, bbox_dir, sdf_dir, class_map_dir, split_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=8)

    for batch in tqdm(dataloader):
        a = 0
