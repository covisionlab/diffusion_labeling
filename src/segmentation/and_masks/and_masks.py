import glob
import numpy as np
import cv2
import torchshow as ts


def compute_and_mask(mask_folder, kind):
    """
    Computes the intersection of a mask and its corresponding bounding boxes,
    creating a refined mask where only pixels inside bounding boxes are preserved. 
    Used in the paper for the experment with downstream task

    Parameters:
        mask_folder (str): Path to the folder containing mask images.
        kind (str): Type of data (e.g., 'layout', 'sdf').
    """
    mask_files = sorted(glob.glob(f"{mask_folder}/*.png"))
    bbox_files = [fpath.replace(f"synthetic/{kind}/masks", "real/bbox").replace('.png', '.txt') for fpath in mask_files]
    assert len(mask_files) == len(bbox_files), f"{len(mask_files)} != {len(bbox_files)}"

    for i, (mask_fpath, bbox_fpath) in enumerate(zip(mask_files, bbox_files)):
        
        bboxes = np.atleast_2d(np.loadtxt(bbox_fpath))
        mask = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE)
        masked_mask = np.zeros_like(mask)

        for cls in range(1, 6):
            cls_mask = (mask == cls)
            bbox_roi = np.zeros_like(mask)
            for row in range(bboxes.shape[0]):
                c, top_left_x, top_left_y, bottom_right_x,  bottom_right_y = bboxes[row]
                if c == cls:
                    c, top_left_x, top_left_y, bottom_right_x,  bottom_right_y = bboxes[row]
                    top_left_x = int(top_left_x * mask.shape[1])
                    top_left_y = int(top_left_y * mask.shape[0])
                    bottom_right_x = int(bottom_right_x * mask.shape[1])
                    bottom_right_y = int(bottom_right_y * mask.shape[0])
                    bbox_roi[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
        
                    anded_mask = bbox_roi * cls_mask
                    masked_mask[anded_mask == 1] = cls
                    # ts.save([cls_mask, bbox_roi, anded_mask, masked_mask], f"_torchshow/lol/{cls}_{out_fpath.split('/')[-1]}")

        out_fpath = mask_fpath.replace("/synthetic/", "/synthetic_masked/")
        # assert cv2.imwrite(out_fpath, masked_mask), f"Error saving {out_fpath}"
        print(f"Saved {out_fpath} that has {masked_mask.sum()} pixels")


if __name__ == "__main__":
    
    print("-------------Layout-------------")
    folder = "/mnt/NAS20/quality/paper_benchmarks/wood_defect_detection/syndata4cv_cvpr25/synthetic/layout/masks"
    compute_and_mask(folder, kind="layout")
    
    print("-------------SDF-------------")
    folder = "/mnt/NAS20/quality/paper_benchmarks/wood_defect_detection/syndata4cv_cvpr25/synthetic/ours/masks"
    compute_and_mask(folder, kind="ours")
