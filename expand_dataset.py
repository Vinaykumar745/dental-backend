from bing_image_downloader import downloader
import os
import shutil

print("Starting dataset expansion...")

queries = [
    {"query": "aphthous ulcer mouth medical", "dir": "aphthous_ulcer"},
    {"query": "erythroplakia oral cavity", "dir": "erythroplakia"},
    {"query": "leukoplakia mouth medical", "dir": "leukoplakia"},
    {"query": "oral lichen planus mouth", "dir": "lichen_planus"},
    {"query": "healthy mouth teeth gums tongue medical", "dir": "normal"},
    {"query": "oral cancer mouth carcinoma", "dir": "oral_cancer"},
    {"query": "oral submucous fibrosis medical", "dir": "oral_submucous_fibrosis"}
]

limit = 100
base_dataset_dir = "dataset"
temp_download_dir = "temp_downloads"

if not os.path.exists(base_dataset_dir):
    os.makedirs(base_dataset_dir)

for q in queries:
    print(f"\n--- Downloading {limit} images for {q['dir']} ---")
    downloader.download(
        q["query"],
        limit=limit,
        output_dir=temp_download_dir,
        adult_filter_off=False,
        force_replace=False,
        timeout=60,
        verbose=False
    )
    
    # Move from temp to actual dataset folder
    downloaded_folder = os.path.join(temp_download_dir, q["query"])
    target_folder = os.path.join(base_dataset_dir, q["dir"])
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        
    if os.path.exists(downloaded_folder):
        for filename in os.listdir(downloaded_folder):
            src_file = os.path.join(downloaded_folder, filename)
            # Find a unique name
            base, ext = os.path.splitext(filename)
            counter = 1
            dst_file = os.path.join(target_folder, filename)
            while os.path.exists(dst_file):
                dst_file = os.path.join(target_folder, f"{base}_{counter}{ext}")
                counter += 1
            shutil.move(src_file, dst_file)
            
print("\nDataset expansion complete!")
