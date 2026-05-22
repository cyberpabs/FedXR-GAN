import tensorflow as tf

from constants import TF_DEVICE, EPOCHS
from dataset_download import download_dataset
from model import DCGAN
from dataset_preprocessing import (
    generate_dataset,
    print_dataset_stats,
    store_preprocessed_images,
    compare_jitter_images,
)

if __name__ == '__main__':
    download_dataset()

    with tf.device(TF_DEVICE):
        dataset = generate_dataset()

        # Show dataset stats and store some preprocessed images
        print_dataset_stats()
        store_preprocessed_images(dataset)
        compare_jitter_images(3)

        # Train the DCGAN model
        dcgan = DCGAN()
        dcgan.train(dataset, EPOCHS)
