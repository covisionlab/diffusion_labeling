import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import pandas as pd
import cv2
from torch.utils.data import DataLoader
from src.segmentation.core.open_wood_dset import OpenWoodSegmentationDataset
from src.segmentation.core.transform_factory import transform_factory

def binary_balance(input_df, retain_class):
    """ Balance the dataset by keeping 10% of non-defected images """

    defected_filepaths = []
    non_defected_filepaths = []
    for (mask_fpath, image_fpath) in zip(input_df['mask'].values, input_df['image'].values):
        mask = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE) == retain_class
        if mask.sum() > 0:
            defected_filepaths.append((image_fpath, mask_fpath))
        else:
            non_defected_filepaths.append((image_fpath, mask_fpath))

    print(f"Defected: {len(defected_filepaths)}, Non-defected: {len(non_defected_filepaths)}")
    
    # create a dataset with 90% defect and 10% non-defect
    ten_percent_no_defect = int(len(defected_filepaths) * 0.1)
    non_defected_filepaths = non_defected_filepaths[:ten_percent_no_defect]

    print(f"[ BALANCED ] Defected: {len(defected_filepaths)}, Non-defected: {len(non_defected_filepaths)}")

    # create a dataframe with img_path, mask_path
    df = pd.DataFrame(defected_filepaths + non_defected_filepaths, columns=['image', 'mask'])
    return df

def get_splits(opt):

    if opt.data.second_train_df is not None:
        train_df = pd.read_csv(opt.data.train_df)
        second_train_df = pd.read_csv(opt.data.second_train_df)
        val_df = pd.read_csv(opt.data.val_df)
        return train_df, second_train_df, val_df

    train_df = pd.read_csv(opt.data.train_df)
    val_df = pd.read_csv(opt.data.val_df)
    return train_df, None, val_df


def data_factory(opt):

    stats = opt.data.stats.mu, opt.data.stats.std
    train_df, second_train_df, val_df = get_splits(opt)
    
    train_df = binary_balance(train_df, opt.data.retain_class)
    if second_train_df is not None:
        second_train_df = binary_balance(second_train_df, opt.data.retain_class)
        train_df = pd.concat([train_df, second_train_df], ignore_index=True)
    
    train_transform, val_transform = transform_factory(opt)


    ###### TRAIN ######
    ###################
    train_dset = OpenWoodSegmentationDataset(
        data_df=train_df,
        retain_class_id=opt.data.retain_class,
        stats=stats, 
        transform=train_transform)
    
    train_dset.use_subset(opt.main.n_train_images)
    train_loader = DataLoader(train_dset, batch_size=opt.hyp.batch_size, shuffle=True, drop_last=False)


    ###### VAL ######
    #################
    val_dset = OpenWoodSegmentationDataset(
        data_df=val_df,
        retain_class_id=opt.data.retain_class,
        stats=stats,
        transform=val_transform)
    
    val_dset.use_subset(opt.main.n_val_images)
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, drop_last=False)


    return train_dset, val_dset, train_loader, val_loader

