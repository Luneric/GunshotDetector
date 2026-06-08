import kagglehub
import shutil
from pathlib import Path

DATASETS = [
    {
        "handle": "zackyzac/firearms-audio-dataset-58-gun-types/versions/1",
        "dest": "data/raw/firearms_58"
    },
    {
        "handle": "omendrakumarupadhyay/gunshot-clean-noisy-dataset",
        "dest": "data/raw/clean_noisy"
    },
    {
        "handle": "emrahaydemr/gunshot-audio-dataset",
        "dest": "data/raw/emrah_gunshot"
    },
    {
        "handle": "mmoreaux/environmental-sound-classification-50",
        "dest": "data/raw/non_gunshot"
    },
    {
        "handle": "chrisfilo/urbansound8k",
        "dest": "data/raw/urban_sound"
    },
]

for ds in DATASETS:
    dest = Path(ds["dest"])
    
    if dest.exists():
        print(f"Skipping {ds['handle']} — already exists at {dest}")
        continue
    
    print(f"\nDownloading {ds['handle']}...")
    path = kagglehub.dataset_download(ds["handle"])
    
    shutil.copytree(path, dest)
    print(f"Copied to: {dest}")

print("\nAll datasets ready.")