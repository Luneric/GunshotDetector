import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
DATASETS = {
    "firearms_58": {"path": "data/raw/firearms_58/dataset", "source": "firearms_58"},
    "clean_noisy":  {"path": "data/raw/clean_noisy",         "source": "clean_noisy"},
    "emrah":        {"path": "data/raw/emrah_gunshot",       "source": "emrah"},
    "non_gunshot":  {"path": "data/raw/non_gunshot/audio/audio", "source": "non_gunshot"},
    "urban_sound": {"path": "data/raw/urban_sound", "source": "urban_sound"},
}

OUT_DIR = "data/processed/spectrograms"
MANIFEST_PATH = "data/manifest.csv"

SAMPLE_RATE = 22050
DURATION = 2.0
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# LABELING RULES
# -------------------------------------------------------

def label_firearms_58(filepath: Path) -> dict:
    # all gunshots, gun type = parent folder name (e.g. ak-12)
    gun_type = filepath.parent.name
    return {"label": "gunshot", "gun_type": gun_type}

def label_clean_noisy(filepath: Path) -> dict:
    name = filepath.stem.lower()
    noise_type = "clean" if "clean" in name else "noisy"
    return {"label": "gunshot", "gun_type": f"unknown_{noise_type}"}

def label_emrah(filepath: Path) -> dict:
    # parent folder is gun type e.g. "AK-12", "MP5"
    gun_type = filepath.parent.name
    return {"label": "gunshot", "gun_type": gun_type}

def label_non_gunshot(filepath: Path) -> dict:
    # ESC-50: last number in filename is category, all are non-gunshot
    return {"label": "no_gunshot", "gun_type": "none"}

def label_urban_sound(filepath: Path) -> dict:
    # filename format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav
    # class 6 = gun_shot
    parts = filepath.stem.split("-")
    class_id = int(parts[1])
    if class_id == 6:
        return {"label": "gunshot", "gun_type": "urban_gunshot"}
    return {"label": "no_gunshot", "gun_type": "none"}

LABELERS = {
    "firearms_58": label_firearms_58,
    "clean_noisy":  label_clean_noisy,
    "emrah":        label_emrah,
    "non_gunshot":  label_non_gunshot,
    "urban_sound": label_urban_sound,
}

# -------------------------------------------------------
# AUDIO PROCESSING
# -------------------------------------------------------

def load_and_pad(filepath):
    y, sr = librosa.load(str(filepath), sr=SAMPLE_RATE, mono=True)
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    else:
        y = y[:TARGET_SAMPLES]
    return y

def to_melspectrogram(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max)

# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------

records = []
failed = []

for ds_name, ds_config in DATASETS.items():
    ds_path = Path(ds_config["path"])
    labeler = LABELERS[ds_config["source"]]

    wav_files = list(ds_path.rglob("*.wav"))
    print(f"\n{ds_name}: {len(wav_files)} files found")

    for wav_file in tqdm(wav_files, desc=ds_name):
        try:
            y = load_and_pad(wav_file)
            mel = to_melspectrogram(y)
            meta = labeler(wav_file)

            out_name = f"{ds_name}__{wav_file.stem}.npy"
            out_path = os.path.join(OUT_DIR, out_name)
            np.save(out_path, mel)

            records.append({
                "spectrogram_path": out_path,
                "original_wav": str(wav_file),
                "label": meta["label"],
                "gun_type": meta["gun_type"],
                "dataset": ds_name,
            })

        except Exception as e:
            failed.append({"file": str(wav_file), "error": str(e)})

# -------------------------------------------------------
# SAVE MANIFEST
# -------------------------------------------------------

df = pd.DataFrame(records)
df.to_csv(MANIFEST_PATH, index=False)

print(f"\n{'='*50}")
print(f"Processed:  {len(df)} files")
print(f"Failed:     {len(failed)} files")
print(f"\nLabel distribution:")
print(df["label"].value_counts())
print(f"\nBy dataset:")
print(df.groupby(['dataset', 'label']).size().unstack(fill_value=0))

if failed:
    pd.DataFrame(failed).to_csv("data/failed_files.csv", index=False)
    print(f"\nFailed files saved to data/failed_files.csv")