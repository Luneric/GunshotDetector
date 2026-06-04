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
]

for ds in DATASETS:
    print(f"\nDownloading {ds['handle']}...")
    path = kagglehub.dataset_download(ds["handle"])
    print(f"Downloaded to cache: {path}")
    
    dest = Path(ds["dest"])
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(path, dest)
    print(f"Copied to: {dest}")

print("\nAll datasets ready.")