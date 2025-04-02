import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A


class OpenWoodSegmentationDataset(Dataset):
    def __init__(self, data_df, retain_class_id, stats, transform):
        
        self.n_classes = 5
        self.data_df = data_df
        self.transform = transform
        self.retain_class_id = retain_class_id
        
        self.mu = torch.tensor(stats[0], requires_grad=False).view(3,1,1)
        self.std = torch.tensor(stats[1], requires_grad=False).view(3,1,1)

    def __getitem__(self, idx):
        img_fpath = self.data_df.iloc[idx]['image']
        mask_fpath = self.data_df.iloc[idx]['mask']

        image = cv2.imread(img_fpath, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE)

        # Remove all the classes except the retain_class_id
        mask = (mask == self.retain_class_id).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # normalization by hand
        image = image / 255.0
        image = torch.from_numpy(image).permute(2,0,1).float()
        image = (image - self.mu) / self.std

        # Convert to 1-hot tensor
        one_hot_mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, one_hot_mask


    def use_subset(self, n):
        """ This function is used to reduce the size of the dataset, can be called after the dataset is created"""
        # take only the first n samples
        self.data_df = self.data_df.iloc[:n]


    def unnorm(self, img):
        """ Remove normalization for visualization purposes """        
        return img * self.std.to(img.device) + self.mu.to(img.device)

    def __len__(self):
        return len(self.data_df)


if __name__ == '__main__':
    import glob
    import pandas as pd

    # retrieve files
    image_fpaths = sorted(glob.glob('/home/fpelosin/projects/misc/open-wood-dset/processed_data/images/*.jpg'))
    mask_fpaths = sorted(glob.glob('/home/fpelosin/projects/misc/open-wood-dset/processed_data/masks/*.png'))

    # create splits
    data_df = pd.DataFrame({'image': image_fpaths, 'mask': mask_fpaths})

    # Create the dataset
    dataset = OpenWoodSegmentationDataset(data_df, 
                                            retain_class_id=3,
                                            stats=( [0.682849358341924, 0.582127860882468, 0.3922592927143617],
                                                    [0.258069769627407, 0.21367024957898295, 0.14352231782814462]),
                                            transform=A.Compose([
                                                    A.Emboss(p=0.3),
                                                    A.GaussNoise(std_range=(0.09, 0.14), p=0.3),
                                                    A.Resize(128, 128),]),
                                                    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Example to loop through the DataLoader
    for images, masks in dataloader:
        print(images.shape, masks.shape)
