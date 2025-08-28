import os
import pandas as pd
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import argparse
from tqdm import tqdm

def copy_images_from_csv(csv_path, output_folder):
    """
    Reads a CSV file containing image file paths and copies them to the specified output folder.
    Skips copying if the file already exists.
    """
    df = pd.read_csv(csv_path)
    images = df['image'].values  # Assumes column 'image' contains file paths

    os.makedirs(output_folder, exist_ok=True)
    
    for img_fpath in images:
        if not os.path.exists(img_fpath):
            print(f"Warning: {img_fpath} does not exist.")
            continue
        
        out_fpath = os.path.join(output_folder, os.path.basename(img_fpath))
        if os.path.exists(out_fpath):
            print(f"Skipping {out_fpath}, already exists.")
            continue
        
        img = cv2.imread(img_fpath, cv2.IMREAD_COLOR)
        cv2.imwrite(out_fpath, img)
        print(f"Saved {out_fpath}")

def compute_metrics(folder_real, folder_synth, batch_size=32, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset1 = ImageFolder(folder_real, transform=transform)
    dataset2 = ImageFolder(folder_synth, transform=transform)
    
    dataloader_real = DataLoader(dataset1, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_synth = DataLoader(dataset2, batch_size=batch_size, shuffle=False, num_workers=4)
    
    fid = MemorizationInformedFrechetInceptionDistance(feature=2048, normalize=True).to(device) 
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
    lpips_scores = []
    
    with torch.no_grad():
        for (images_real, _), (images_synth, _) in tqdm(zip(dataloader_real, dataloader_synth), total=len(dataloader_real)):
            images_real, images_synth = images_real.to(device), images_synth.to(device)
            fid.update(images_real, real=True)
            fid.update(images_synth, real=False)
            lpips_scores.append(lpips(images_real, images_synth).mean().item())
    
    fid_score = fid.compute().item()
    lpips_score = sum(lpips_scores) / len(lpips_scores)
    
    return fid_score, lpips_score

def main():
    parser = argparse.ArgumentParser(description="Process CSVs, copy images, and compute metrics.")
    parser.add_argument("csv_real", type=str, help="Path to the CSV file with real images")
    parser.add_argument("csv_synth", type=str, help="Path to the CSV file with synthetic images")
    parser.add_argument("output_folder", type=str, help="Path to the output folder where images will be saved")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing images")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda)")
    
    args = parser.parse_args()
    folder_real = os.path.join(args.output_folder, "real")
    folder_synth = os.path.join(args.output_folder, "synth")
    
    # Create an additional subfolder inside real/0 for torchmetrics
    folder_real_sub = os.path.join(folder_real, "0")
    os.makedirs(folder_real_sub, exist_ok=True)

    # Create an additional subfolder inside synth/0 for torchmetrics
    folder_synth_sub = os.path.join(folder_synth, "0")
    os.makedirs(folder_synth_sub, exist_ok=True)

    copy_images_from_csv(args.csv_real, folder_real_sub)
    copy_images_from_csv(args.csv_synth, folder_synth_sub)
    
    fid_score, lpips_score = compute_metrics(folder_real, folder_synth, args.batch_size, args.device)
    print(f"FID Score: {fid_score:.4f}")
    print(f"LPIPS Score: {lpips_score:.4f}")

if __name__ == "__main__":
    main()
