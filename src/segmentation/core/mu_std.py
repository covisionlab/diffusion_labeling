import numpy as np
import glob
import cv2
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def process_image(img_file):
    """Process a single image and return its statistics"""
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    stats = {
        'mu_r': np.mean(img[:,:,0]),
        'mu_g': np.mean(img[:,:,1]),
        'mu_b': np.mean(img[:,:,2]),
        'std_r': np.std(img[:,:,0]),
        'std_g': np.std(img[:,:,1]),
        'std_b': np.std(img[:,:,2]),
        'count': 1
    }
    
    return stats

def combine_stats(stats_list):
    """Combine statistics from multiple images"""
    total_stats = {
        'mu_r': 0, 'mu_g': 0, 'mu_b': 0,
        'std_r': 0, 'std_g': 0, 'std_b': 0,
        'count': 0
    }
    
    for stats in stats_list:
        for key in total_stats:
            total_stats[key] += stats[key]
    
    count = total_stats['count']
    for key in total_stats:
        if key != 'count':
            total_stats[key] /= count
    
    return total_stats

def main():
    data_folders = [
        "/home/fpelosin/projects/misc/open-wood-dset/processed_data/images",
    ]
    
    # Determine number of CPU cores to use
    num_cores = mp.cpu_count() - 1  # Leave one core free
    if num_cores < 1:
        num_cores = 1
    
    print(f"Using {num_cores} CPU cores for processing")
    
    for data_folder in data_folders:
        files = glob.glob(data_folder + '/*.jpg', recursive=True)
        total_files = len(files)
        
        if total_files == 0:
            print(f"No files found in {data_folder}")
            continue
            
        print(f"Processing {total_files} files in {data_folder}")
        
        # Create a pool of workers
        with mp.Pool(processes=num_cores) as pool:
            # Process images in parallel and collect results
            results = list(tqdm(
                pool.imap(process_image, files),
                total=total_files,
                desc="Processing images"
            ))
        
        # Combine all results
        combined_stats = combine_stats(results)
        
        # Print results
        print(data_folder)
        print(f"({combined_stats['mu_r']/255}, {combined_stats['mu_g']/255}, {combined_stats['mu_b']/255})")
        print(f"({combined_stats['std_r']/255}, {combined_stats['std_g']/255}, {combined_stats['std_b']/255})")

if __name__ == "__main__":
    main()