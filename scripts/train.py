import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

MANIFEST_PATH = "data/manifest.csv"
MODEL_OUT      = "models/stage1_gunshot_detector.keras"
LOGS_DIR       = "models/logs"

BATCH_SIZE  = 128
EPOCHS      = 30
LR          = 1e-3
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
RANDOM_SEED = 42

INPUT_SHAPE = (128, 87, 1)

os.makedirs("models", exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

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
# TF DATASET PIPELINE
# -------------------------------------------------------

def load_spectrogram(path, label):
    def _load(p):
        p = p.numpy().decode("utf-8")
        mel = np.load(p).astype(np.float32)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        mel = mel[..., np.newaxis]
        return mel
    mel = tf.py_function(_load, [path], tf.float32)
    mel.set_shape(INPUT_SHAPE)
    return mel, label

def make_dataset(df, shuffle=False):
    paths  = df["spectrogram_path"].values
    labels = df["binary_label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED)
    ds = ds.map(load_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(df_train, shuffle=True)
val_ds   = make_dataset(df_val)
test_ds  = make_dataset(df_test)

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------

def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Classifier head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="stage1_gunshot_detector")

model = build_model(INPUT_SHAPE)
model.summary()

# -------------------------------------------------------
# COMPILE
# -------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="binary_crossentropy",
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
        filepath=MODEL_OUT,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR),
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
plt.savefig("models/training_curves.png", dpi=150)
print("\nSaved training_curves.png")
print(f"Model saved to {MODEL_OUT}")