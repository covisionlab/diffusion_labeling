import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from const import ID_to_NAME_and_COLOR

# This script visualizes for debugging purposes the processed data by overlaying segmentation masks and bounding boxes on the original images.

if __name__ == "__main__":

    # Set the root directories
    raw_data_folder = "raw_data"
    processed_data_folder = "processed_data"

    # Create directories for processed data
    os.makedirs(f"{processed_data_folder}/viz", exist_ok=True)

    for i, img_name in enumerate(glob.glob(f"{processed_data_folder}/images/*.jpg")):
        # Retrieve a random image
        # img_name = np.random.choice(glob.glob("processed_data/images/*.jpg"))
        img_id = (img_name.split("/")[-1]).split(".")[0]
        print("Processing:", img_id)

        # Visualize the processed data
        img_filename = f"{processed_data_folder}/images/{img_id}.jpg"
        img = cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB)

        # Read and superimpose mask
        mask_filename = f"{processed_data_folder}/masks/{img_id}.png"
        img_segm = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

        # Load original mask (Semantic Maps)
        original_mask_filename = f"{raw_data_folder}/Semantic Maps/{img_id}_segm.bmp"
        original_mask = cv2.cvtColor(cv2.imread(original_mask_filename), cv2.COLOR_BGR2RGB)

        # Read bounding boxes in YOLO format
        bbox_filename = f"{processed_data_folder}/bbox/{img_id}.txt"
        bbx = np.loadtxt(bbox_filename).reshape(-1, 5)  # YOLO format: class x_center y_center width height

        # Overlay mask on image
        overlay = img.copy()
        overlay[img_segm > 0] = [255, 0, 0]  # Example: Red overlay for segmentation mask
        alpha = 0.5  # Transparency
        blended_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Draw bounding boxes on the image with label
        # the format is label_id top_left_x top_left_y bottom_right_x bottom_right_y
        bbox_img = blended_img.copy()
        img_h, img_w, _ = img.shape
        for row in bbx:
            label_id, top_left_x, top_left_y, bottom_right_x, bottom_right_y = row
            top_left_x = int(top_left_x * img_w)
            top_left_y = int(top_left_y * img_h)
            bottom_right_x = int(bottom_right_x * img_w)
            bottom_right_y = int(bottom_right_y * img_h)
            bbox_img = cv2.rectangle(bbox_img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
            bbox_img = cv2.putText(bbox_img, ID_to_NAME_and_COLOR[int(label_id)][0], (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        # Display the results side-by-side
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(original_mask)
        axs[1].set_title("Original Mask (Semantic Map)")
        axs[1].axis("off")

        axs[2].imshow(bbox_img)
        axs[2].set_title("Processed Data (Overlay + BBox)")
        axs[2].axis("off")

        plt.tight_layout()
        plt.savefig(f"{processed_data_folder}/viz/{i}.png")
        plt.close()