# References:
# https://www.kaggle.com/code/peremartramanonellas/gan-tutorial-first-dcgan-using-tensorflow
# https://www.tensorflow.org/tutorials/generative/dcgan
# https://www.kaggle.com/code/chiragsethi50/dcgan
# https://www.kaggle.com/code/kaledhoshme/using-gan-to-generate-chest-x-ray-images
# https://www.kaggle.com/code/asif00/medical-image-generation-with-gans
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose

import tensorflow as tf 
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import time

from constants import (
    BATCH_SIZE,
    NOISE_DIM,
    NUM_EXAMPLES_TO_GENERATE,
    PREPROCESSED_IMAGE_SIZE,
    CHECKPOINT_DIR,
)

class DCGAN():
    def __init__(self, batch_size = BATCH_SIZE, noise_dim = NOISE_DIM, checkpoint_dir = CHECKPOINT_DIR):
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.image_size = PREPROCESSED_IMAGE_SIZE
        self.seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, self.noise_dim])

        self.generator_filters = {
            (64, 64): [256, 128, 64, 32, 16],
            (128, 128): [256, 128, 64, 32, 16, 8],
        }
        self.discriminator_filters = {
            (64, 64): [64, 128, 256],
            (128, 128): [64, 128, 256, 512],
        }

        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()

        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def make_generator_model(self):
        filters = self.generator_filters[self.image_size]

        model = tf.keras.Sequential()

        model.add(layers.Dense(8*8*filters[0], use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((8, 8, filters[0])))
        assert model.output_shape == (None, 8, 8, filters[0])

        model.add(layers.Conv2DTranspose(filters[1], (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, filters[1])
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        for f in filters[2:]:
            model.add(layers.Conv2DTranspose(f, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.image_size[0], self.image_size[1], 1)

        return model

    def make_discriminator_model(self):
        filters = self.discriminator_filters[self.image_size]

        model = tf.keras.Sequential()
        model.add(layers.Conv2D(filters[0], (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[self.image_size[0], self.image_size[1], 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        for f in filters[1:]:
            model.add(layers.Conv2D(f, (5, 5), strides=(2, 2), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_function(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_function(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self, fake_output):
        return self.loss_function(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def generate_images(self, model, output_path='generated_image.png'):
        test_input = tf.random.normal([1, self.noise_dim])
        predictions = model(test_input, training=False)

        image = predictions[0, :, :, 0] * 127.5 + 127.5  # Rescale to [0, 255]
        plt.imsave(output_path, image.numpy(), cmap='gray')
        print(f"Image saved to {output_path}")

    def generate_and_save_images(self, model, epoch, test_input, output_dir = './training_images'):
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'image_at_epoch_{:07d}.png'.format(epoch)))
        
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            self.generate_and_save_images(self.generator, epoch + 1, self.seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        self.generate_and_save_images(self.generator, epochs, self.seed)

    def restore_model_from_checkpoint(self, checkpoint_dir = CHECKPOINT_DIR, save_model = True):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        if latest_checkpoint:
            status = self.checkpoint.restore(latest_checkpoint)
            status.assert_existing_objects_matched()  # Optional: asserts that all objects were restored
            print("Checkpoint restored from:", latest_checkpoint)
        else:
            print("No checkpoint found in", checkpoint_dir)

        if save_model:
            self.generator.save('generator_model.keras')

        return self.generator