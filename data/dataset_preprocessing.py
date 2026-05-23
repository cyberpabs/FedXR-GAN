# See: https://www.tensorflow.org/tutorials/load_data/images

import os
import tensorflow as tf
import matplotlib.pyplot as plt

from data.constants import (
    BUFFER_SIZE, 
    BATCH_SIZE, 
    DATASET_DATA,
    PREPROCESSED_IMAGE_SIZE,
)

def print_dataset_stats(dataset_option = 'chest-xray-pneumonia'):
    data_dir = DATASET_DATA[dataset_option]['path']
    list_ds = tf.data.Dataset.list_files(os.path.join(data_dir, f'*.{DATASET_DATA[dataset_option]["extension"]}'), shuffle=True)

    print("Total files:", tf.data.experimental.cardinality(list_ds).numpy())
    print("\nExample file paths:")
    for f in list_ds.take(5):
        print(f.numpy())

    print("\nInspecting raw bytes for a single file:")
    for file_tensor in list_ds.take(1):
        file_path = file_tensor.numpy().decode("utf-8")
        img_bytes = tf.io.read_file(file_path)

        # Print the filepath and the first 100 bytes
        print("File path:", file_path)
        print("Raw bytes (first 100):", img_bytes.numpy()[:100])


def apply_jitter(image):
    # Apply random brightness, random contrast and random horizontal flip
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_flip_left_right(image)
    
    return image

def process_image(file_path, jitter = False):
    # Read the image file as bytes
    img_bytes = tf.io.read_file(file_path)

    # Use channels = 1 for grayscale scans
    img_decoded = tf.image.decode_jpeg(img_bytes, channels=1)

    # Resize the image to a fixed size (e.g., 128x128) for consistency in training
    # NOTE: the original image size varies on each datapoint on 'chest-xray-pneumonia' dataset
    img_resized = tf.image.resize(img_decoded, PREPROCESSED_IMAGE_SIZE) 
    img_scaled = img_resized / 255.0
    
    if jitter:
        # Apply jitter to the image
        img_scaled = apply_jitter(img_scaled)
        
        # Ensure pixel values are still in [0, 1]
        img_scaled = tf.clip_by_value(img_scaled, 0.0, 1.0)

    # Normalize the pixel values to [-1, 1]
    img_normalized = (img_scaled * 2.0) - 1.0
    return img_normalized

def generate_dataset(ds_option = 'chest-xray-pneumonia', jitter = False):
    # 1. Create a dataset object
    data_dir = DATASET_DATA[ds_option]['path']
    list_ds = tf.data.Dataset.list_files(os.path.join(data_dir, f'*.{DATASET_DATA[ds_option]["extension"]}'), shuffle=True)

    # 2. Build the dataset with preprocessing
    dataset = list_ds.map(lambda x: process_image(x, jitter), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

def store_preprocessed_images(dataset, output_path = "dataset_images/preprocessed_images.png"):
    for batch in dataset.take(1):
        plt.figure(figsize=(8, 8))

        for i in range(9): # Show 9 images
            plt.subplot(3, 3, i + 1)
            # Convert from [-1, 1] back to [0, 1] for display
            img = (batch[i] * 0.5) + 0.5

            # Remove the grayscale channel dimension (so shape becomes PREPROCESS_IMAGE_SIZE)
            img = tf.squeeze(img, axis=-1)

            img_np = img.numpy()
            plt.imshow(img_np, cmap='gray')
            plt.axis("off")

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path)


def compare_jitter_images(n, dataset_option = 'chest-xray-pneumonia', output_path="dataset_images/compare_jitter.png"):
    data_dir = DATASET_DATA[dataset_option]['path']
    list_ds = tf.data.Dataset.list_files(os.path.join(data_dir, f'*.{DATASET_DATA[dataset_option]["extension"]}'), shuffle=True)

    # Gather 'n' file paths as a list of Python strings
    file_paths = list_ds.take(n)
    file_path_list = list(file_paths.as_numpy_iterator())  # convert dataset to a Python list of tensors
    file_path_list = [fp.decode('utf-8') for fp in file_path_list]

    # Process each file path with and without jitter
    images_no_jitter = []
    images_jitter = []
    
    for fp in file_path_list:
        images_no_jitter.append(process_image(fp, jitter=False))
        images_jitter.append(process_image(fp, jitter=True))

    # Plot them side-by-side ('n' rows, 2 columns -> left = no jitter, right = jitter)
    plt.figure(figsize=(6, 3*n))

    for i in range(n):
        # Convert from [-1, 1] back to [0, 1] for display
        img_nj = (images_no_jitter[i] * 0.5) + 0.5
        img_j  = (images_jitter[i] * 0.5) + 0.5

        # Remove grayscale channel dimension
        img_nj = tf.squeeze(img_nj, axis=-1)
        img_j  = tf.squeeze(img_j, axis=-1)

        nj_np = img_nj.numpy()
        j_np  = img_j.numpy()

        # Left subplot: no jitter
        ax1 = plt.subplot(n, 2, 2*i + 1)
        plt.imshow(nj_np, cmap='gray')
        plt.title("No Jitter")
        plt.axis("off")

        # Right subplot: jitter
        ax2 = plt.subplot(n, 2, 2*i + 2)
        plt.imshow(j_np, cmap='gray')
        plt.title("With Jitter")
        plt.axis("off")

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == '__main__':
    print_dataset_stats()
    ds = generate_dataset()
    store_preprocessed_images(ds)
    compare_jitter_images(3)
