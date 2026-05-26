# task.py
# Defines data loading per client partition and re-exports the model factory.
# Each client gets a stratified subset of the training images based on partition_id.
# Images are interleaved by class before splitting so each client sees both classes.
# Val split is also stratified to avoid class imbalance issues.

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from data.constants import BATCH_SIZE, PREPROCESSED_IMAGE_SIZE, DATASET_OPTIONS
from models.classifier import make_classifier, load_test

TRAIN_DIR = os.path.join(DATASET_OPTIONS["chest-xray-pneumonia"]["path"], "chest_xray", "train")
TEST_DIR = os.path.join(DATASET_OPTIONS["chest-xray-pneumonia"]["path"], "chest_xray", "test")

IMG_H, IMG_W = PREPROCESSED_IMAGE_SIZE
VAL_SPLIT = 0.2
SEED = 42

def load_partition(partition_id, num_partitions):
    # Collect paths per class separately
    classes = ["NORMAL", "PNEUMONIA"]
    class_paths = {cls: [] for cls in classes}

    for cls in classes:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                class_paths[cls].append(os.path.join(cls_dir, fname))

    # Interleave NORMAL and PNEUMONIA so each partition gets both classes
    normal_pairs = [(p, 0.0) for p in class_paths["NORMAL"]]
    pneumonia_pairs = [(p, 1.0) for p in class_paths["PNEUMONIA"]]
    
    # Split normal and pneumonia paths into chunks for each partition manually
    n_chunk = len(normal_pairs) // num_partitions
    p_chunk = len(pneumonia_pairs) // num_partitions
    
    # Slice a proportional slice of both classes for this specific client partition ID
    n_start, n_end = partition_id * n_chunk, (partition_id + 1) * n_chunk
    p_start, p_end = partition_id * p_chunk, (partition_id + 1) * p_chunk
    
    # Handle the final partition's leftovers cleanly
    if partition_id == num_partitions - 1:
        n_end = len(normal_pairs)
        p_end = len(pneumonia_pairs)
        
    partition_pairs = normal_pairs[n_start:n_end] + pneumonia_pairs[p_start:p_end]
    partition_paths = [p for p, _ in partition_pairs]
    partition_labels = [l for _, l in partition_pairs]

    if len(partition_paths) == 0:
        raise ValueError(f"Partition {partition_id} is empty. Reduce num-partitions or num-supernodes.")

    print(f"Client {partition_id}: {len(partition_paths)} images (indices {n_start} to {n_end})")

    # Separate by class for stratified val split
    normal_paths  = [(p, l) for p, l in zip(partition_paths, partition_labels) if l == 0.0]
    pneumonia_paths = [(p, l) for p, l in zip(partition_paths, partition_labels) if l == 1.0]

    val_n = int(len(normal_paths) * VAL_SPLIT)
    val_p = int(len(pneumonia_paths) * VAL_SPLIT)

    val_pairs   = normal_paths[:val_n] + pneumonia_paths[:val_p]
    train_pairs = normal_paths[val_n:] + pneumonia_paths[val_p:]

    def make_ds(pairs, shuffle=True):
        paths  = [p for p, _ in pairs]
        labels = [l for _, l in pairs]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(pairs), seed=SEED)
        return ds

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

    train_ds = (make_ds(train_pairs, shuffle=True)
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))

    val_ds = (make_ds(val_pairs, shuffle=False)
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds

# Re-export for server_app and client_app to import from one place
def load_model():
    return make_classifier()

def load_test_data():
    return load_test(TEST_DIR)