# References:
# https://www.tensorflow.org/tutorials/images/classification
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

import os
import time
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from data.constants import (
    BATCH_SIZE,
    PREPROCESSED_IMAGE_SIZE,
    DATASET_OPTIONS,
)

TRAIN_DIR = os.path.join(DATASET_OPTIONS["chest-xray-pneumonia"]["path"], "chest_xray", "train")
TEST_DIR = os.path.join(DATASET_OPTIONS["chest-xray-pneumonia"]["path"], "chest_xray", "test")
VAL_DIR = os.path.join(DATASET_OPTIONS["chest-xray-pneumonia"]["path"], "chest_xray", "val")

CLASSIFIER_CHECKPOINT_DIR = "./checkpoints/classifier_checkpoints"
CLASSIFIER_RESULTS_DIR = "./results/classifier_results"

IMG_H, IMG_W = PREPROCESSED_IMAGE_SIZE
EPOCHS = 100


# Dataset

VAL_SPLIT = 0.2
SEED = 42

def load_train_val(directory):
    """80/20 split from train directory."""
    common = dict(
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        seed=SEED,
        validation_split=VAL_SPLIT,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory, subset="training", **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory, subset="validation", **common
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        return image, label

    def augment_fn(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.clip_by_value(image, -1.0, 1.0)
        return image, label

    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def load_test(directory):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
    )
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 127.5 - 1.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_and_wrap_model(model_path):
    """Loads a saved model and recompiles it for evaluation."""
    model = tf.keras.models.load_model(model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# Model

def make_classifier():
    model = tf.keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", input_shape=(IMG_H, IMG_W, 1)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.MaxPooling2D(),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.MaxPooling2D(),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.MaxPooling2D(),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1),
    ])
    return model


# Training

class Classifier:
    def __init__(self):
        self.model = make_classifier()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(3e-5),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

    def train(self, train_ds, val_ds, epochs=EPOCHS):
        os.makedirs(CLASSIFIER_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(CLASSIFIER_RESULTS_DIR, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(CLASSIFIER_CHECKPOINT_DIR, "best.keras"),
                monitor="val_auc",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=8, mode="max", restore_best_weights=True, verbose=1
            ),
        ]

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )

        self._plot_history(history)
        return history

    def evaluate(self, test_ds):
        results = self.model.evaluate(test_ds, return_dict=True)
        print("\nTest Results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

        # Compute F1 from precision + recall
        p = results.get("precision", 0)
        r = results.get("recall", 0)
        f1 = 2 * p * r / (p + r + 1e-8)
        print(f"  f1: {f1:.4f}")

        # Threshold tuning
        self._tune_threshold(test_ds)
        return results

    def _tune_threshold(self, test_ds):
        import numpy as np

        all_logits, all_labels = [], []
        for x, y in test_ds:
            logits = self.model(x, training=False)
            all_logits.append(logits.numpy())
            all_labels.append(y.numpy())

        logits  = np.concatenate(all_logits).flatten()
        labels  = np.concatenate(all_labels).flatten()
        probs   = 1 / (1 + np.exp(-logits))  # sigmoid

        print("\nThreshold Sweep: ")
        print(f"  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.1, 0.95, 0.05):
            preds = (probs >= thresh).astype(float)
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()
            acc  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            print(f"  {thresh:>10.2f} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        print(f"\n  Best threshold: {best_thresh:.2f} → F1: {best_f1:.4f}")

    def save(self, path="classifier_model.keras"):
        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M%S")
        stamped = path.replace(".keras", f"_{ts}.keras")
        self.model.save(stamped)
        print(f"Model saved to {stamped}")

    def _plot_history(self, history):
        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M%S")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.history["loss"], label="train")
        axes[0].plot(history.history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()

        axes[1].plot(history.history["auc"], label="train")
        axes[1].plot(history.history["val_auc"], label="val")
        axes[1].set_title("AUC")
        axes[1].legend()

        plt.tight_layout()
        out = os.path.join(CLASSIFIER_RESULTS_DIR, f"training_curves_{ts}.png")
        plt.savefig(out)
        print(f"Training curves saved to {out}")

if __name__ == "__main__":
    import argparse
    from gan.constants import TF_DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", type=str, default=None,
                        help="Path to saved model to wrap and evaluate without training")
    args = parser.parse_args()

    with tf.device(TF_DEVICE):
        test_ds = load_test(TEST_DIR)

        if args.evaluate_only:
            model = load_and_wrap_model(args.evaluate_only)
            clf = Classifier.__new__(Classifier)
            clf.model = model
            clf.evaluate(test_ds)
        else:
            train_ds, val_ds = load_train_val(TRAIN_DIR)

            clf = Classifier()
            clf.model.summary()
            clf.train(train_ds, val_ds)
            clf.evaluate(test_ds)
            clf.save()