import tensorflow as tf

from gan.constants import TF_DEVICE, EPOCHS
from gan.model import DCGAN
from data.dataset_download import download_dataset
from data.dataset_preprocessing import (
    generate_dataset,
    print_dataset_stats,
    store_preprocessed_images,
    compare_jitter_images,
)

if __name__ == '__main__':
    download_dataset('chest-xray-pneumonia')

    with tf.device(TF_DEVICE):
        dataset = generate_dataset(jitter=True)

        # Show dataset stats and store some preprocessed images
        print_dataset_stats()
        store_preprocessed_images(dataset)
        compare_jitter_images(3)

        # Train the DCGAN model
        dcgan = DCGAN()
        dcgan.train(dataset, EPOCHS)
