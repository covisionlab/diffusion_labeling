import glob
import numpy as np
import cv2


def compute_merror(mask_folder,kind):
    """
    Computes hit/miss pixels. Used to compute SAE in the paper.

    Parameters:
        mask_folder (str): Path to the folder containing mask images.
        kind (str): Type of dataset (e.g., 'layout', 'sdf').
    """

    mask_files = sorted(glob.glob(f"{mask_folder}/*.png"))
    bbox_files = [fpath.replace(f"synthetic/{kind}/masks", "real/bbox").replace('.png', '.txt') for fpath in mask_files]
    assert len(mask_files) == len(bbox_files), f"{len(mask_files)} != {len(bbox_files)}"

    cls_error = {cls: 0 for cls in range(1, 7)}
    cls_hits = {cls: 0 for cls in range(1, 7)}
    cls_pixels = {cls: 0 for cls in range(1, 7)}
    tot_error = 0
    tot_hits = 0
    tot_pixels = 0
    for mask_fpath, bbox_fpath in zip(mask_files, bbox_files):
        
        bboxes = np.atleast_2d(np.loadtxt(bbox_fpath))
        mask = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE)

        for cls in range(1, 7):
            mask_cls = (mask == cls)
            error_mask = np.ones_like(mask_cls)
            
            if bboxes.size != 0:
                for row in range(bboxes.shape[0]):
                    c, top_left_x, top_left_y, bottom_right_x,  bottom_right_y = bboxes[row]
                    if c == cls:
                        c, top_left_x, top_left_y, bottom_right_x,  bottom_right_y = bboxes[row]
                        top_left_x = int(top_left_x * mask.shape[1])
                        top_left_y = int(top_left_y * mask.shape[0])
                        bottom_right_x = int(bottom_right_x * mask.shape[1])
                        bottom_right_y = int(bottom_right_y * mask.shape[0])
                        error_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

            error_mask = error_mask * mask_cls
            hit_mask = np.bitwise_not(error_mask) * mask_cls
            cls_error[cls] += error_mask.sum()
            cls_hits[cls] += hit_mask.sum()
            cls_pixels[cls] += mask_cls.sum()

            tot_error += error_mask.sum()
            tot_hits += hit_mask.sum()
            tot_pixels += mask_cls.sum()


    for cls in range(1, 7):
        print(f"[Class {cls}] err: {cls_error[cls]:<10}  pixels: {cls_pixels[cls]:<10}  hits: {cls_hits[cls]:<10}  merror: {cls_error[cls]/cls_pixels[cls]:.4f}  mhit: {cls_hits[cls]/cls_pixels[cls]:.4f}")
    print(f"[Total] err: {tot_error:<10}  pixels: {tot_pixels:<10}  hits: {tot_hits:<10}  merror: {tot_error/tot_pixels:.4f}  mhit: {tot_hits/tot_pixels:.4f}") 


if __name__ == "__main__":
    print("-------------Layout-------------")
    folder = "synthetic/layout/masks"
    compute_merror(folder, kind="layout")
    
    print("-------------SDF-------------")
    folder = "synthetic/ours/masks"
    compute_merror(folder, kind="ours")

