import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np

batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128

#Preprocess Data Here

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

discriminator = keras.Sequential(
    [
        layers.InputLayer((28, 28, discriminator_in_channels)),
        layers.Conv2D(64, (3,3), strides=(2,2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3,3), strides=(2,2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

generator = keras.Sequential(
    [
        layers.InputLayer((generator_in_channels,)),
        layers.Dense(7 * 7 * generator_in_channels),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, generator_in_channels)),
        layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(conditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

        @property
        def metrics(self):
            return [self.gen_loss_tracker, self.disc_loss_tracker]

        def compile(self, d_optimizer, g_optimizer, loss_fn):
            super(ConditionalGAN, self).compile()
            self.d_optimizer = d_optimizer
            self.g_optimizer = g_optimizer
            self.loss_fn = loss_fn

        def train_step(self, data):
            real_images, one_hot_labels = data

            #For discriminator
            image_one_hot_labels = one_hot
