## Overview

> Note:
The default empty bbox is 6 0 0 0 0 this is because of layout diffusion, ask Alessandro Simoni for more information. Also, we use scaling factor 8 and the paddings to have a 256 x 256 images for the diffusion, since the original images are very.

This script processes images and their corresponding annotations of the Wood dataset which can be found here [https://zenodo.org/records/4694695#.YkWqTX9Bzmg](https://zenodo.org/records/4694695#.YkWqTX9Bzmg).

## How to run

Pay attention to the directory structure before running. Make sure you downloaded the dataset. Then:

```bash
python preprocess.py
```

## Directory Structure

The script assumes the following directory structure:
- `raw_data/` - Root directory for source data
  - `Images*/` - Contains original BMP images
  - `Bouding Boxes/` - Contains bounding box annotations (filename format: `{image_id}_anno.txt`)
  - `Semantic Maps/` - Contains segmentation masks (filename format: `{image_id}_segm.bmp`)

- `processed_data/` - Destination directory for processed data
  - `images/` - Processed and resized images (JPG format)
  - `masks/` - Processed segmentation masks (PNG format)
  - `bbox/` - Processed bounding box coordinates (TXT format)
  - `viz/` - Appears to be for visualization (unused in main code)

## Key Constants

- `SCALING_FACTOR = 8` - Factor by which images are downsampled
- `LEFT_PAD = RIGHT_PAD = 8` - Padding added to the left and right sides of images
- `YOLO_FORMAT = False` - Flag to determine bounding box format (corner vs. YOLO center format)

## Main Functions

### `process_image(inp)`

The core function that processes a single image and its annotations. It takes a tuple containing:
- `img_name` - Path to the image file
- `n` - Current image index
- `tot` - Total number of images

Processing steps:
1. Extract the image ID from the filename
2. Skip images with incorrect dimensions (expected width: 2800px)
3. Read and sanitize bounding box annotations (fixing comma/dot and case formatting issues)
4. Handle empty annotation files by creating empty masks and default bounding boxes
5. Pad the original image to a fixed size (2816×1024) and resize by `SCALING_FACTOR`
6. Save the processed image to the destination directory
7. Load and process the corresponding segmentation mask
8. For each annotation:
   - Extract the label and coordinates
   - Map old label IDs to new IDs
   - Skip certain labels (overgrown or unknown)
   - Adjust bounding box coordinates for padding
   - Format coordinates according to YOLO or corner format
   - Update the segmentation mask using the color information
9. Save the bounding box annotations to a text file
10. Resize and save the segmentation mask

### `reset_dst_dir()`

Resets the destination directory by removing it and recreating all necessary subdirectories.

## Parallel Processing

The script uses `joblib.Parallel` to process images in parallel, utilizing all available CPU cores (with `n_jobs=-1`).

## Special Notes

1. The script handles several data formatting issues in the original dataset:
   - Commas instead of dots in coordinate values
   - Inconsistent case in label names
   - Images with incorrect dimensions
   - Empty annotation files

2. The script filters out certain labels (IDs 10, 6, and 11, referred to as "overgrown" and unknown)

3. The script uses a fixed padding and scaling approach to standardize all images

4. The script converts between coordinate systems, adjusting for the added padding


## Error Handling

The script includes basic error handling for:
- Empty annotation files
- Image saving failures
- Images with incorrect dimensions

## Output Format

1. Images: JPGs resized by factor of 8
2. Segmentation masks: PNGs with label IDs as pixel values
3. Bounding boxes: Text files with either corner coordinates or YOLO format (depending on `YOLO_FORMAT` flag)