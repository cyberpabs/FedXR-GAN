import tensorflow as tf
from model import DCGAN

dcgan = DCGAN()
model = tf.keras.models.load_model('generator_model_128px.keras')
dcgan.generate_images(model, 'restore_model_image_example.png')