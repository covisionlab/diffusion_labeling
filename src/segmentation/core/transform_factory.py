import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import cv2
import albumentations as A


def transform_factory(opt):
    
    # [NOTE] all the normalizations are done in the dataset class

    # resize requested for the model
    resize_h = opt.data.augmentation.resize_h
    resize_w = opt.data.augmentation.resize_w
    original_h = opt.data.original_height
    original_w = opt.data.original_width

    if resize_h is None or resize_w is None:
        resize_h = original_h
        resize_w = original_w

    s = [

            # /////////// 0 ///////////
            A.Compose([
            A.Resize(resize_h, resize_w),
            ]),

            # /////////// 1 ///////////
            A.Compose([
                A.Emboss(p=0.3),
                A.GaussNoise(std_range=(0.09, 0.14), p=0.3),
                A.Flip(p=0.3),
                # A.Resize(resize_h, resize_w),
            ]),
            # /////////// 2  ///////////
            # (same as 1 but with higher probabilities)
            A.Compose([
            A.RandomCrop(height=int(original_h - original_h * 0.25), 
                            width=int(original_w - original_w * 0.25), p=0.4),
            A.Resize(resize_h, resize_w),
            A.OneOf([A.HorizontalFlip(p=0.4), A.VerticalFlip(p=0.4)]),
            A.ShiftScaleRotate(shift_limit=0.09, 
                                scale_limit=0.09, 
                                rotate_limit=20, 
                                border_mode=cv2.BORDER_CONSTANT, 
                                value=0, 
                                mask_value=0, 
                                p=0.4),
            A.Emboss(p=0.4),
            ]),
        ]

    aug_train = s[opt.data.augmentation.severity]

    aug_val = A.Compose([A.Resize(resize_h, resize_w)])

    return aug_train, aug_val

