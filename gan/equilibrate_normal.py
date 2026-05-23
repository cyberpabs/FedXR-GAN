import os
import tensorflow as tf

from data.constants import BATCH_SIZE, DATASET_OPTIONS
from gan.constants import NOISE_DIM

TRAIN_DIR = os.path.join(DATASET_OPTIONS["chest-xray-pneumonia"]["path"], "chest_xray", "train")
GENERATOR_PATH = os.path.join(os.path.dirname(__file__), "generator_model_128px.keras")

def count_images(path):
    return len([f for f in os.listdir(path) if f.lower().endswith((".jpeg", ".jpg", ".png"))])

def equilibrate_normal(generator_path=GENERATOR_PATH, train_dir=TRAIN_DIR):
    normal_dir = os.path.join(train_dir, "NORMAL")
    pneumonia_dir = os.path.join(train_dir, "PNEUMONIA")
    missing = count_images(pneumonia_dir) - count_images(normal_dir)

    if missing <= 0:
        print("NORMAL already has at least as many images as PNEUMONIA.")
        return 0

    generator = tf.keras.models.load_model(generator_path)
    saved = 0
    start = len([f for f in os.listdir(normal_dir) if f.startswith("generated_normal_")])
    while saved < missing:
        batch_size = min(BATCH_SIZE, missing - saved)
        noise = tf.random.normal([batch_size, NOISE_DIM])
        images = generator(noise, training=False)
        images = tf.cast((images + 1.0) * 127.5, tf.uint8)

        for image in images:
            output_path = os.path.join(normal_dir, f"generated_normal_{start + saved:05d}.jpeg")
            tf.io.write_file(output_path, tf.io.encode_jpeg(image))
            saved += 1

    print(f"Generated {saved} NORMAL images in {normal_dir}")
    return saved


if __name__ == "__main__":
    equilibrate_normal()
