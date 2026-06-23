import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------
# HARDWARE SETUP
# -------------------------------------------------------
print("Checking for available GPUs and configuring memory footprint...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}. Enforcing experimental memory growth limits.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected. Training will run on CPU.")

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
BASE_DIR      = os.path.expanduser("~/research/ELP_Research/GunshotDetector")
MANIFEST_PATH = os.path.join(BASE_DIR, "data/model2_threat_manifest.csv")
MODEL_OUT     = os.path.join(BASE_DIR, "models/threat_classifier_best.keras")
LOGS_DIR      = os.path.join(BASE_DIR, "models/logs_model2")
PLOTS_DIR     = os.path.join(BASE_DIR, "models")

BATCH_SIZE       = 64
EPOCHS           = 50
LR               = 1e-4      # FIX: was 3e-5 — too slow, val loss diverged before model learned
VAL_SPLIT        = 0.20
TEST_SPLIT       = 0.10
RANDOM_SEED      = 42
NORM_SAMPLE_SIZE = 2000

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# -------------------------------------------------------
# LOAD & SPLIT
# -------------------------------------------------------
print(f"\nLoading manifest from: {MANIFEST_PATH}")
df = pd.read_csv(MANIFEST_PATH)
df["full_spectrogram_path"] = df["spectrogram_path"].apply(
    lambda x: os.path.join(BASE_DIR, x)
)

print(f"Total samples: {len(df)}")
print(df["threat_level"].value_counts())

df_train, df_temp = train_test_split(
    df, test_size=(VAL_SPLIT + TEST_SPLIT),
    stratify=df["threat_level"], random_state=RANDOM_SEED
)
df_val, df_test = train_test_split(
    df_temp, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
    stratify=df_temp["threat_level"], random_state=RANDOM_SEED
)

print(f"\nTrain: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
print("\nVal class breakdown:")
print(df_val["threat_level"].value_counts())
print("\nTest class breakdown:")
print(df_test["threat_level"].value_counts())

# -------------------------------------------------------
# CLASS WEIGHTS
# -------------------------------------------------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=df_train["threat_level"].values
)
class_weights = {0: weights[0], 1: weights[1]}
print(f"\nClass weights: {class_weights}")

# -------------------------------------------------------
# GLOBAL NORMALIZATION STATS
# -------------------------------------------------------
print("\nComputing global normalization stats from training set sample...")
sample_paths = (
    df_train["spectrogram_path"]
    .sample(min(NORM_SAMPLE_SIZE, len(df_train)), random_state=RANDOM_SEED)
    .apply(lambda x: os.path.join(BASE_DIR, x))
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
# TF DATASET PIPELINE
# -------------------------------------------------------
def load_spectrogram(file_path, label):
    def _load(p):
        p = p.numpy().decode("utf-8")
        mel = np.load(p).astype(np.float32)
        mel = (mel - GLOBAL_MEAN) / GLOBAL_STD
        if len(mel.shape) == 2:
            mel = np.expand_dims(mel, axis=-1)
        return mel
    mel = tf.py_function(_load, [file_path], tf.float32)
    mel.set_shape([None, None, 1])
    label = tf.cast(label, tf.int32)
    return mel, label

def make_dataset(df, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((
        df["full_spectrogram_path"].values,
        df["threat_level"].values
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED)
    ds = ds.map(load_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(BATCH_SIZE, padded_shapes=([None, None, 1], []))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(df_train, shuffle=True)
val_ds   = make_dataset(df_val)
test_ds  = make_dataset(df_test)

# Detect dynamic input shape from first batch
for sample_x, _ in train_ds.take(1):
    input_shape = sample_x.shape[1:]
    print(f"\nDetected input shape: {input_shape}")

# -------------------------------------------------------
# MODEL
# FIX: Removed Block 3 — model was too large, overfit by epoch 3
# FIX: Replaced GlobalAveragePooling with Avg+Max concatenation
#      to capture more discriminative frequency features
# -------------------------------------------------------
def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # FIX: Dual pooling — captures both average energy and peak frequency features
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    # Classifier head
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # 0=Small, 1=Large

    return tf.keras.Model(inputs, outputs, name="threat_classifier_v2")

model = build_model(input_shape)
model.summary()

# -------------------------------------------------------
# COMPILE
# FIX: Removed label_smoothing — hurts learning when boundary is already unclear
# -------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR, weight_decay=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),  # FIX: no label_smoothing
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]
)

# -------------------------------------------------------
# CALLBACKS
# FIX: EarlyStopping now monitors val_loss (not val_auc)
#      Val loss was diverging from epoch 3 while val_auc kept climbing —
#      monitoring val_auc let training continue 37 extra epochs for nothing.
# FIX: Added second EarlyStopping on val_loss as safety net
# -------------------------------------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_OUT,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    # Primary stopper: halt when val_loss starts diverging
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=6,               # FIX: was monitoring val_auc with patience=10
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR),
]

# -------------------------------------------------------
# TRAIN
# -------------------------------------------------------
print("\nStarting Model 2 training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# -------------------------------------------------------
# EVALUATE ON HELD-OUT TEST SET
# -------------------------------------------------------
print("\nEvaluating on test set...")
results = model.evaluate(test_ds, verbose=1)
metrics = dict(zip(model.metrics_names, results))
print("\nTest Results:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# Collect predictions
print("\nCollecting predictions for diagnostic plots...")
y_true, y_pred_prob = [], []
for batch_x, batch_y in test_ds:
    preds = model.predict(batch_x, verbose=0)
    y_pred_prob.extend(preds.flatten().tolist())
    y_true.extend(batch_y.numpy().tolist())

y_true      = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred      = (y_pred_prob >= 0.5).astype(int)

# Classification report
print("\n" + "="*55)
print("         THREAT SPLIT EVALUATION REPORT")
print("="*55)
print(classification_report(
    y_true, y_pred,
    target_names=["0: Small Firearm", "1: Large Firearm"],
    zero_division=0
))

# PR-AUC
pr_auc = average_precision_score(y_true, y_pred_prob)
print(f"PR-AUC (average precision): {pr_auc:.4f}")

# Confusion matrix printout
cm = confusion_matrix(y_true, y_pred)
print(f"\nConfusion Matrix:")
print(cm)
print(f"  True Negatives  (Small correctly identified) : {cm[0][0]}")
print(f"  False Positives (Small misclassified as Large): {cm[0][1]}")
print(f"  False Negatives (Large misclassified as Small): {cm[1][0]}")
print(f"  True Positives  (Large correctly identified) : {cm[1][1]}")
print("="*55)

# -------------------------------------------------------
# PLOT 1: Training curves (loss + AUC)
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(history.history["loss"],     label="Train Loss",  color="royalblue",  lw=2)
axes[0].plot(history.history["val_loss"], label="Val Loss",    color="darkorange", lw=2)
axes[0].set_title("Loss Profile Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Crossentropy Loss")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].plot(history.history["auc"],     label="Train AUC", color="royalblue",  lw=2)
axes[1].plot(history.history["val_auc"], label="Val AUC",   color="darkorange", lw=2)
axes[1].set_title("Separation Capacity Curve (ROC AUC)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("AUC")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
curves_path = os.path.join(PLOTS_DIR, "threat_training_curves.png")
plt.savefig(curves_path, dpi=150)
plt.close()
print(f"\nSaved threat_training_curves.png → {curves_path}")

# -------------------------------------------------------
# PLOT 2: Precision-Recall Curve + PR-AUC
# -------------------------------------------------------
precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_prob)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall_vals, precision_vals, lw=2, color="royalblue", label=f"PR-AUC = {pr_auc:.4f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (Test Set)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
pr_path = os.path.join(PLOTS_DIR, "threat_pr_curve.png")
plt.savefig(pr_path, dpi=150)
plt.close()
print(f"Saved threat_pr_curve.png → {pr_path}  (PR-AUC = {pr_auc:.4f})")

# -------------------------------------------------------
# PLOT 3: Confusion Matrix
# -------------------------------------------------------
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Small Firearm", "Large Firearm"]
)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix (Test Set, threshold = 0.5)")
plt.tight_layout()
cm_path = os.path.join(PLOTS_DIR, "threat_confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved threat_confusion_matrix.png → {cm_path}")

# -------------------------------------------------------
# PLOT 4: Precision & Recall vs Threshold
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, precision_vals[:-1], label="Precision", color="royalblue",  lw=2)
ax.plot(thresholds, recall_vals[:-1],    label="Recall",    color="darkorange", lw=2)
ax.axvline(x=0.5, color="gray", linestyle="--", label="threshold = 0.5")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision & Recall vs Decision Threshold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
thresh_path = os.path.join(PLOTS_DIR, "threat_threshold_curve.png")
plt.savefig(thresh_path, dpi=150)
plt.close()
print(f"Saved threat_threshold_curve.png → {thresh_path}")

# -------------------------------------------------------
# SUMMARY
# -------------------------------------------------------
tn, fp, fn, tp = cm.ravel()
print(f"\n{'='*55}")
print(f"SUMMARY")
print(f"{'='*55}")
print(f"  Test samples                 : {len(y_true)}")
print(f"  PR-AUC                       : {pr_auc:.4f}")
print(f"  True Positives  (Large)      : {tp}")
print(f"  False Negatives (Large→Small): {fn}")
print(f"  False Positives (Small→Large): {fp}")
print(f"  True Negatives  (Small)      : {tn}")
print(f"\nAll outputs saved to: {PLOTS_DIR}")
print(f"  threat_training_curves.png")
print(f"  threat_pr_curve.png          (PR-AUC = {pr_auc:.4f})")
print(f"  threat_confusion_matrix.png")
print(f"  threat_threshold_curve.png")
print(f"  {os.path.basename(MODEL_OUT)}")