import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

# -------------------------------------------------------
# CONFIG — must match exactly what was used in training
# -------------------------------------------------------
BASE_PROJECT_DIR = "/home/naxiong/research/ELP_Research/GunshotDetector"
CAROLYN_OUT_DIR  = os.path.join(BASE_PROJECT_DIR, "carolynscriptt")

MANIFEST_PATH = os.path.join(BASE_PROJECT_DIR, "data/manifest.csv")
MODEL_PATH     = os.path.join(CAROLYN_OUT_DIR, "stage1_gunshot_detector.keras")

BATCH_SIZE   = 64
VAL_SPLIT    = 0.20   # must match training split to get the same test set
TEST_SPLIT   = 0.10
RANDOM_SEED  = 42
INPUT_SHAPE  = (128, 87, 1)

# Number of training files to sample for global normalization stats
NORM_SAMPLE_SIZE = 2000

# -------------------------------------------------------
# LOAD & REPRODUCE THE SAME SPLIT AS TRAINING
# -------------------------------------------------------
df = pd.read_csv(MANIFEST_PATH)
df["binary_label"] = (df["label"] == "gunshot").astype(int)

print(f"Total samples: {len(df)}")
print(df["label"].value_counts())

df_train, df_temp = train_test_split(
    df, test_size=(VAL_SPLIT + TEST_SPLIT),
    stratify=df["binary_label"], random_state=RANDOM_SEED
)
df_val, df_test = train_test_split(
    df_temp, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
    stratify=df_temp["binary_label"], random_state=RANDOM_SEED
)

print(f"\nTrain: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
print("\nTest class breakdown:")
print(df_test["binary_label"].value_counts())

# -------------------------------------------------------
# GLOBAL NORMALIZATION STATS
# Must be computed the same way as during training
# -------------------------------------------------------
print("\nComputing global normalization stats from training set sample...")
sample_paths = (
    df_train["spectrogram_path"]
    .sample(min(NORM_SAMPLE_SIZE, len(df_train)), random_state=RANDOM_SEED)
    .apply(lambda x: os.path.join(BASE_PROJECT_DIR, x))
    .values
)

all_vals = []
for p in sample_paths:
    try:
        mel = np.load(p).astype(np.float32)
        all_vals.append(mel.ravel())
    except Exception as e:
        print(f"  Warning: could not load {p}: {e}")

all_vals    = np.concatenate(all_vals)
GLOBAL_MEAN = float(all_vals.mean())
GLOBAL_STD  = float(all_vals.std()) + 1e-8
print(f"  Global mean: {GLOBAL_MEAN:.4f} | Global std: {GLOBAL_STD:.4f}")

# -------------------------------------------------------
# DATASET PIPELINE
# -------------------------------------------------------
def load_spectrogram(path, label):
    def _load(p):
        p = p.numpy().decode("utf-8")
        mel = np.load(p).astype(np.float32)
        mel = (mel - GLOBAL_MEAN) / GLOBAL_STD
        mel = mel[..., np.newaxis]
        return mel
    mel = tf.py_function(_load, [path], tf.float32)
    mel.set_shape(INPUT_SHAPE)
    return mel, label

def make_dataset(df):
    paths  = df["spectrogram_path"].apply(lambda x: os.path.join(BASE_PROJECT_DIR, x)).values
    labels = df["binary_label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

test_ds = make_dataset(df_test)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
print(f"\nLoading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# -------------------------------------------------------
# EVALUATE (built-in keras metrics)
# -------------------------------------------------------
print("\nEvaluating on test set...")
results = model.evaluate(test_ds, verbose=1)
metrics = dict(zip(model.metrics_names, results))
print("\nTest Results:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# -------------------------------------------------------
# COLLECT PREDICTIONS
# -------------------------------------------------------
print("\nCollecting predictions...")
y_true, y_pred_prob = [], []
for batch_x, batch_y in test_ds:
    preds = model.predict(batch_x, verbose=0)
    y_pred_prob.extend(preds.flatten().tolist())
    y_true.extend(batch_y.numpy().tolist())

y_true      = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred      = (y_pred_prob >= 0.5).astype(int)

# -------------------------------------------------------
# CLASSIFICATION REPORT
# -------------------------------------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["non-gunshot", "gunshot"]))

# -------------------------------------------------------
# PR-AUC
# -------------------------------------------------------
pr_auc = average_precision_score(y_true, y_pred_prob)
print(f"PR-AUC (average precision): {pr_auc:.4f}")

# -------------------------------------------------------
# PLOT: PR Curve
# -------------------------------------------------------
precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_prob)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall_vals, precision_vals, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (Test Set)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
pr_path = os.path.join(CAROLYN_OUT_DIR, "eval_pr_curve.png")
plt.savefig(pr_path, dpi=150)
plt.close()
print(f"Saved eval_pr_curve.png → {pr_path}")

# -------------------------------------------------------
# PLOT: Confusion Matrix
# -------------------------------------------------------
cm   = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-gunshot", "gunshot"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix (Test Set, threshold = 0.5)")
plt.tight_layout()
cm_path = os.path.join(CAROLYN_OUT_DIR, "eval_confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved eval_confusion_matrix.png → {cm_path}")

# -------------------------------------------------------
# PLOT: Precision & Recall vs Threshold
# (helps you pick a better threshold if needed)
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, precision_vals[:-1], label="Precision", lw=2)
ax.plot(thresholds, recall_vals[:-1],    label="Recall",    lw=2)
ax.axvline(x=0.5, color="gray", linestyle="--", label="threshold = 0.5")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision & Recall vs Decision Threshold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
thresh_path = os.path.join(CAROLYN_OUT_DIR, "eval_threshold_curve.png")
plt.savefig(thresh_path, dpi=150)
plt.close()
print(f"Saved eval_threshold_curve.png → {thresh_path}")

# -------------------------------------------------------
# SUMMARY
# -------------------------------------------------------
print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"  Test samples     : {len(y_true)}")
print(f"  PR-AUC           : {pr_auc:.4f}")
tn, fp, fn, tp = cm.ravel()
print(f"  True Positives   : {tp}  (gunshots correctly caught)")
print(f"  False Negatives  : {fn}  (gunshots missed)")
print(f"  False Positives  : {fp}  (false alarms)")
print(f"  True Negatives   : {tn}  (non-gunshots correctly rejected)")
print(f"\nOutputs saved to: {CAROLYN_OUT_DIR}")
print(f"  eval_pr_curve.png")
print(f"  eval_confusion_matrix.png")
print(f"  eval_threshold_curve.png")
