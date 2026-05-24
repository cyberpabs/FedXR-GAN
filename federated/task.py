# task.py
# Defines data loading per client partition and re-exports the model factory.
# Each client gets a subset of the training images based on partition_id.

import sys, os
# Fix: Python doesn't see the root, hence this line
# LEAVE IT HERE UP
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#print(sys.path[0])


import tensorflow as tf
from data.constants import BATCH_SIZE, PREPROCESSED_IMAGE_SIZE, DATASET_OPTIONS
from models.classifier import make_classifier, load_test

TRAIN_DIR = os.path.join(DATASET_OPTIONS["chest-xray-pneumonia"]["path"], "chest_xray", "train")
TEST_DIR = os.path.join(DATASET_OPTIONS["chest-xray-pneumonia"]["path"], "chest_xray", "test")

IMG_H, IMG_W = PREPROCESSED_IMAGE_SIZE
VAL_SPLIT = 0.2
SEED = 42

def load_partition(partition_id, num_partitions):
    # Collect all image paths and labels from NORMAL and PNEUMONIA subdirs
    classes = ["NORMAL", "PNEUMONIA"]
    all_paths = []
    all_labels = []

    for label, cls in enumerate(classes):
        cls_dir = os.path.join(TRAIN_DIR, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                all_paths.append(os.path.join(cls_dir, fname))
                all_labels.append(float(label))

    total = len(all_paths)

    # Split paths into equal chunks, one per client
    chunk = total // num_partitions
    start = partition_id * chunk
    end = start + chunk if partition_id < num_partitions - 1 else total

    partition_paths = all_paths[start:end]
    partition_labels = all_labels[start:end]

    print(f"Client {partition_id}: {len(partition_paths)} images (indices {start} to {end})")

    # Build a tf.data pipeline from the selected paths
    path_ds = tf.data.Dataset.from_tensor_slices((partition_paths, partition_labels))
    path_ds = path_ds.shuffle(buffer_size=len(partition_paths), seed=SEED)

    # Split into train and validation
    val_size = int(len(partition_paths) * VAL_SPLIT)
    val_ds = path_ds.take(val_size)
    train_ds = path_ds.skip(val_size)

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, (IMG_H, IMG_W))
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.clip_by_value(img, -1.0, 1.0)
        return img, label

    train_ds = (train_ds
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))

    val_ds = (val_ds
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds

# Re-export for server_app and client_app to import from one place
def load_model():
    return make_classifier()

def load_test_data():
    return load_test(TEST_DIR)
