# See: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/ 
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm

from data.constants import DATASET_DATA
from data.dataset_preprocessing import process_image

# Inception V3 model for feature extraction
inception_model = tf.keras.applications.InceptionV3(
    include_top=False, 
    pooling='avg', 
    input_shape=(128, 128, 3)
)

def calculate_fid(real_images, generated_images):
    """
    Computes the FID score using the globally loaded inception_model.
    """
    # Extract features using the global model
    act1 = inception_model.predict(real_images)
    act2 = inception_model.predict(generated_images)
    
    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # Calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    # Calculate and return the final FID score
    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


if __name__ == "__main__":
    data_dir = DATASET_DATA['chest-xray-pneumonia']['path']
    data_extension = DATASET_DATA['chest-xray-pneumonia']['extension']
    
    all_files = glob.glob(os.path.join(data_dir, f'*.{data_extension}'))
    gen_files = [f for f in all_files if 'generated' in f]
    other_files = [f for f in all_files if 'generated' not in f]

    generated_ds = tf.data.Dataset.from_tensor_slices(gen_files).shuffle(len(gen_files) or 1)
    other_ds = tf.data.Dataset.from_tensor_slices(other_files).shuffle(len(other_files) or 1)
    
    generated_dataset = generated_ds.map(
        lambda x: preprocess_input(tf.image.grayscale_to_rgb(process_image(x))), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    other_dataset = other_ds.map(
        lambda x: preprocess_input(tf.image.grayscale_to_rgb(process_image(x))), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    print(f"generated_files: {len(gen_files)}")
    print(f"generated_files examples: {gen_files[:5]}")
    print(f"other_files: {len(other_files)}")
    print(f"other_files examples: {other_files[:5]}")
    
    fid_score = calculate_fid(generated_dataset.batch(256), other_dataset.batch(256))
    print(f"FID Score: {fid_score}")