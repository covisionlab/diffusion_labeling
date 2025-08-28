import glob
import numpy as np
import cv2


def compute_hit(mask_folder,kind):
    """
    Computes hit/miss statistics for bounding boxes inside mask images. Used to compute EBR in the paper

    Parameters:
        mask_folder (str): Path to the folder containing mask images.
        kind (str): Type of dataset (e.g., 'layout', 'sdf').
    """

    mask_files = sorted(glob.glob(f"{mask_folder}/*.png"))
    bbox_files = [fpath.replace(f"synthetic/{kind}/masks", "real/bbox").replace('.png', '.txt') for fpath in mask_files]
    assert len(mask_files) == len(bbox_files), f"{len(mask_files)} != {len(bbox_files)}"

    bbox_hits = {cls: 0 for cls in range(1, 7)}
    bbox_miss = {cls: 0 for cls in range(1, 7)}
    total_bbox = {cls: 0 for cls in range(1, 7)}
    for mask_fpath, bbox_fpath in zip(mask_files, bbox_files):
        
        bboxes = np.atleast_2d(np.loadtxt(bbox_fpath))
        mask = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE)

        for cls in range(1, 7):
            mask_cls = (mask == cls)
            bbox_mask = np.zeros_like(mask_cls)
            
            if bboxes.size != 0:
                for row in range(bboxes.shape[0]):
                    c, top_left_x, top_left_y, bottom_right_x,  bottom_right_y = bboxes[row]
                    if c == cls:
                        c, top_left_x, top_left_y, bottom_right_x,  bottom_right_y = bboxes[row]
                        top_left_x = int(top_left_x * mask.shape[1])
                        top_left_y = int(top_left_y * mask.shape[0])
                        bottom_right_x = int(bottom_right_x * mask.shape[1])
                        bottom_right_y = int(bottom_right_y * mask.shape[0])
                        bbox_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
                        total_bbox[cls] += 1

                        if (bbox_mask * mask_cls).sum() == 0:
                            bbox_miss[cls] += 1

            if  (bbox_mask * mask_cls).sum() > 0:
                bbox_hits[cls] += 1

    for cls in range(1, 7):
        print(f"[Class {cls}] hits: {bbox_hits[cls]:<10}  total: {total_bbox[cls]:<10}  hit rate: {bbox_hits[cls]/total_bbox[cls]:.4f}")
    print(f"[ Total ] hits: {sum(bbox_hits.values()):<10}  total: {sum(total_bbox.values()):<10}  hit rate: {sum(bbox_hits.values())/sum(total_bbox.values()):.4f}")
    print("-------------------------------------------------")
    for cls in range(1, 7):
        print(f"[Class {cls}] misses: {bbox_miss[cls]:<10}  total: {total_bbox[cls]:<10}  miss rate: {bbox_miss[cls]/total_bbox[cls]:.4f}")
    print(f"[ Total ] misses: {sum(bbox_miss.values()):<10}  total: {sum(total_bbox.values()):<10}  miss rate: {sum(bbox_miss.values())/sum(total_bbox.values()):.4f}")


if __name__ == "__main__":
    print("############ Layout #############")
    print("##################################")
    folder = "synthetic/layout/masks"
    compute_hit(folder, kind="layout")
    
    print("############ SDF #############")
    print("##################################")
    folder = "synthetic/ours/masks"
    compute_hit(folder, kind="ours")
