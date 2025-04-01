from joblib import Parallel, delayed
# preprocess the data
import pandas as pd
import os
import cv2
import glob
import numpy as np
from io import StringIO
from const import ID_REMAPPER,  NAME_to_ID_and_COLOR



def reset_dst_dir():
    os.system(f"rm -rf {dst_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(dst_dir + "/images", exist_ok=True)
    os.makedirs(dst_dir + "/masks", exist_ok=True)
    os.makedirs(dst_dir + "/bbox", exist_ok=True)
    os.makedirs(dst_dir + "/viz", exist_ok=True)
    

def process_image(inp):

    LEFT_PAD = RIGHT_PAD = 8
    SCALING_FACTOR = 8
    YOLO_FORMAT = False
    PADDED_WIDTH = 2800 + LEFT_PAD + RIGHT_PAD  # 2816

    # [ debug ] keep track of processed images
    # with open("/tmp/processed_wood.log", "a") as f:
    #     f.write(img_name + "\n")

    # unpack input
    img_name, n, tot= inp
    
    # concurrent logging
    print(f'{n/tot:.2%}', end='\r')

    # 1. read image
    img_id = (img_name.split("/")[-1]).split(".")[0]
    original_img = cv2.imread(img_name)
    
    # Throws away imgs with a fucked up size (there are some images with strange sizes )
    if original_img.shape[1] != 2800:
        return
    
    # 2. read bbox
    try:
        # Replace commas with dots if there are commas 
        # (there is a "formatting bug" in the original data, sometimes there are commas instead of dots)
        with open(f"{root_dir}/Bouding Boxes/{img_id}_anno.txt", 'r', encoding='utf-8') as file:
            content = file.read()
        updated_content = content.replace(',', '.')
        df = pd.read_csv(StringIO(updated_content), sep="\t", header=None)

        # Sanitize labels to lowercase
        # (there is a "formatting bug" in the original data, sometimes there are uppercase letters for the labels)
        df[0] = df[0].str.lower()
        empty_img = False

    # 3. if no bbox, save empty mask and bbox
    except pd.errors.EmptyDataError:
        cv2.imwrite(f"{dst_dir}/masks/{img_id}.png", np.zeros((1024//SCALING_FACTOR, PADDED_WIDTH//SCALING_FACTOR), dtype=np.uint8))
        with open(f"{dst_dir}/bbox/{img_id}.txt", "w") as f:
            f.write("6 0 0 0 0\n")
        empty_img = True
    
    
    # 4. convert img to fixed size (2800 x 1024), actually it will be (PADDED_WIDTH x 1024) because of the padding
    img = cv2.copyMakeBorder(original_img, 0, 0, LEFT_PAD, RIGHT_PAD, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 5. Save image to dst directory
    new_x, new_y = img.shape[1]//SCALING_FACTOR, img.shape[0]//SCALING_FACTOR
    img = cv2.resize(img, (new_x, new_y), interpolation = cv2.INTER_NEAREST)
    assert cv2.imwrite(f"{dst_dir}/images/{img_id}.jpg", img), f"Error saving image {img_name}"

    # 6. Read mask image
    filename_segm = f"{root_dir}/Semantic Maps/{img_id}_segm.bmp"

    # 7. Convert mask to fixed size (PADDED_WIDTH x 1024)
    img_segm = cv2.cvtColor(cv2.imread(filename_segm), cv2.COLOR_BGR2RGB)
    img_segm = cv2.copyMakeBorder(img_segm, 0, 0, LEFT_PAD, RIGHT_PAD, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    segm_label = np.zeros((1024, PADDED_WIDTH), dtype=np.uint8)

    # 8. For each detection get bbox label and mask
    if not empty_img:
        bboxes = ""
        for row in df.iterrows():

            label, coords = row[1][0], row[1][1:].values.astype(np.float32)
            color, label_id = NAME_to_ID_and_COLOR[label][0], NAME_to_ID_and_COLOR[label][1]

            label_id = ID_REMAPPER[label_id]

            # skip overgrown
            if label_id == 10 or label_id == 6:
                print(f"overgrown, should not be here, {img_id}")
                continue

            # skip 
            if label_id == 11:
                print(f"?????, should not be here, {img_id}")
                continue

            top_left_x, top_left_y, bottom_right_x, bottom_right_y = coords

            # fix bbox coords with pad
            top_left_x = ((top_left_x * 2800) + LEFT_PAD) / PADDED_WIDTH
            bottom_right_x = ((bottom_right_x * 2800) + LEFT_PAD) / PADDED_WIDTH

            # 9. convert bbox coords in YOLO format
            if YOLO_FORMAT:
                x_centre = (top_left_x + bottom_right_x) / 2
                y_centre = (top_left_y + bottom_right_y) / 2
                width = bottom_right_x - top_left_x
                height = bottom_right_y - top_left_y
                bboxes += f"{label_id} {x_centre} {y_centre} {width} {height}\n"
            
            # 9. convert bbox coords in corner format
            else:
                bboxes += f"{label_id} {top_left_x} {top_left_y} {bottom_right_x} {bottom_right_y}\n"

            # 10. update mask for each label
            r, g, b = color
            mask = (
                (img_segm[:, :, 0] == r)
                * (img_segm[:, :, 1] == g)
                * (img_segm[:, :, 2] == b)
            )

            segm_label[mask] = label_id

        # 11. save bbox to dst directory
        with open(f"{dst_dir}/bbox/{img_id}.txt", "w") as f:
            # this check is needed because we throw away overgronw for example
            # and if there are no bboxes left, we need to save an empty bbox
            if bboxes == "":
                f.write("6 0 0 0 0\n")
                print(f"Empty bbox, {img_id}!")
            else:
                f.write(bboxes)

    # 12. save mask to dst dectory as .png
    new_x, new_y = segm_label.shape[1]//SCALING_FACTOR, segm_label.shape[0]//SCALING_FACTOR
    segm_label = cv2.resize(segm_label, (new_x, new_y), interpolation = cv2.INTER_NEAREST)
    assert cv2.imwrite(f"{dst_dir}/masks/{img_id}.png", segm_label)


#-------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # run preprocess of data
    root_dir = "raw_data"
    dst_dir = "processed_data"

    # create directories
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(dst_dir + "/images", exist_ok=True)
    os.makedirs(dst_dir + "/masks", exist_ok=True)
    os.makedirs(dst_dir + "/bbox", exist_ok=True)

    # Filter out images fname in the min_df.csv
    fnames = (glob.glob(root_dir + "/Images*/*.bmp"))

    reset_dst_dir()

    #Use joblib to parallelize the loop
    Parallel(n_jobs=-1)(delayed(process_image)((img_fname, i, len(fnames))) for i, img_fname in enumerate(fnames))

    # [ debug ] check if everything is ok img per img
    # for i, img_name in enumerate(fnames[:100]):
    #    process_image((img_name,i, len(fnames)))