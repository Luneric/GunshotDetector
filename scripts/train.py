import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Force Matplotlib to use a headless backend so it safely saves plots over SSH
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------
# HARDWARE
# -------------------------------------------------------
print("Configuring GPU memory growth")
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
BASE_PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_SUBDIR    = os.environ.get("OUTPUT_SUBDIR", "models")

OUT_DIR        = BASE_PROJECT_DIR / OUTPUT_SUBDIR
MANIFEST_PATH  = BASE_PROJECT_DIR / "data" / "manifest.csv"
MODEL_OUT      = OUT_DIR / "stage1_gunshot_detector.keras"
LOGS_DIR       = OUT_DIR / "logs"
NORM_STATS_OUT = OUT_DIR / "norm_stats.npz"

BATCH_SIZE  = 256
EPOCHS      = 50
LR          = 1e-4          
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
RANDOM_SEED = 42

INPUT_SHAPE = (128, 87, 1)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------------------------------------
# LOAD & SPLIT
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
    df_temp, test_size=0.5,
    stratify=df_temp["binary_label"], random_state=RANDOM_SEED
)

print(f"\nTrain: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

# -------------------------------------------------------
# CLASS WEIGHTS
# -------------------------------------------------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=df_train["binary_label"].values
)
class_weights = {0: weights[0], 1: weights[1]}
print(f"\nClass weights: {class_weights}")

# -------------------------------------------------------
# GLOBAL NORMALIZATION STATS
# -------------------------------------------------------
def resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else BASE_PROJECT_DIR / p

print("\nComputing global normalization stats from training set...")
sample_paths = df_train["spectrogram_path"].apply(resolve_path).values
n_stat_samples = min(500, len(sample_paths))
rng = np.random.default_rng(RANDOM_SEED)
stat_sample = rng.choice(sample_paths, size=n_stat_samples, replace=False)

acc_sum, acc_sq_sum, acc_count = 0.0, 0.0, 0
for p in stat_sample:
    mel = np.load(p).astype(np.float32)
    acc_sum += mel.sum()
    acc_sq_sum += (mel ** 2).sum()
    acc_count += mel.size

GLOBAL_MEAN = acc_sum / acc_count
GLOBAL_STD  = np.sqrt(acc_sq_sum / acc_count - GLOBAL_MEAN ** 2)
print(f"Global mean: {GLOBAL_MEAN:.4f} | Global std: {GLOBAL_STD:.4f}")

np.savez(NORM_STATS_OUT, mean=GLOBAL_MEAN, std=GLOBAL_STD)
print(f"Saved normalization stats to {NORM_STATS_OUT}")

GLOBAL_MEAN_T = tf.constant(GLOBAL_MEAN, dtype=tf.float32)
GLOBAL_STD_T  = tf.constant(GLOBAL_STD, dtype=tf.float32)

# -------------------------------------------------------
# AUGMENTATION
# -------------------------------------------------------
def spec_augment(mel, max_freq_mask=12, max_time_mask=12, n_masks=2):
    mel = tf.identity(mel)
    freq_dim = tf.shape(mel)[0]
    time_dim = tf.shape(mel)[1]

    for _ in range(n_masks):
        f_width = tf.random.uniform([], 0, max_freq_mask, dtype=tf.int32)
        f_start = tf.random.uniform([], 0, tf.maximum(freq_dim - f_width, 1), dtype=tf.int32)
        freq_mask = tf.concat([
            tf.ones([f_start, time_dim, 1], dtype=tf.float32),
            tf.zeros([f_width, time_dim, 1], dtype=tf.float32),
            tf.ones([freq_dim - f_start - f_width, time_dim, 1], dtype=tf.float32),
        ], axis=0)
        mel = mel * freq_mask

        t_width = tf.random.uniform([], 0, max_time_mask, dtype=tf.int32)
        t_start = tf.random.uniform([], 0, tf.maximum(time_dim - t_width, 1), dtype=tf.int32)
        time_mask = tf.concat([
            tf.ones([freq_dim, t_start, 1], dtype=tf.float32),
            tf.zeros([freq_dim, t_width, 1], dtype=tf.float32),
            tf.ones([freq_dim, time_dim - t_start - t_width, 1], dtype=tf.float32),
        ], axis=1)
        mel = mel * time_mask

    mel = mel + tf.random.normal(tf.shape(mel), mean=0.0, stddev=0.05)
    return mel

# -------------------------------------------------------
# TF DATASET PIPELINE
# -------------------------------------------------------
def load_spectrogram(path, label):
    def _load(p):
        p = p.numpy().decode("utf-8")
        p = str(resolve_path(p))
        mel = np.load(p).astype(np.float32)
        mel = mel[..., np.newaxis]
        return mel
    mel = tf.py_function(_load, [path], tf.float32)
    mel.set_shape(INPUT_SHAPE)
    mel = (mel - GLOBAL_MEAN_T) / (GLOBAL_STD_T + 1e-8)
    return mel, label

def make_dataset(df, shuffle=False, augment=False):
    paths  = df["spectrogram_path"].values
    labels = df["binary_label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED)
    ds = ds.map(load_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (spec_augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(df_train, shuffle=True, augment=True)
val_ds   = make_dataset(df_val)
test_ds  = make_dataset(df_test)

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(16, (3,3), padding="same", activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Classifier head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="stage1_gunshot_detector")

model = build_model(INPUT_SHAPE)
model.summary()

# -------------------------------------------------------
# COMPILE
# -------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    # Set label_smoothing to 0.0 so the metrics reflect unwarped confidence thresholds
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]
)

# -------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_OUT),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=8,
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
    tf.keras.callbacks.TensorBoard(log_dir=str(LOGS_DIR)),
]

# -------------------------------------------------------
# TRAIN
# -------------------------------------------------------
print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# -------------------------------------------------------
# EVALUATE
# -------------------------------------------------------
print("\nEvaluating on test set...")
results = model.evaluate(test_ds)
metrics = dict(zip(model.metrics_names, results))
print(f"\nTest Results:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# -------------------------------------------------------
# PLOT
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"],     label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].legend()

axes[1].plot(history.history["auc"],     label="train")
axes[1].plot(history.history["val_auc"], label="val")
axes[1].set_title("AUC")
axes[1].legend()

plt.tight_layout()
plot_out_path = OUT_DIR / "training_curves.png"
plt.savefig(plot_out_path, dpi=150)
print(f"\nSaved training_curves.png to {plot_out_path}")
print(f"Model saved to {MODEL_OUT}")