import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    ReLU,
    LeakyReLU,
    Conv2DTranspose,
    InputLayer,
    Flatten,
    Dense,
)


def mvp_generator():
    """
    Create the minimum Viable product generator to get started
    """
    generator = Sequential(name="mvp_generator")

    # Handle input layer
    generator.add(InputLayer(input_shape=(256,256, 3)))

    # Create the Downsampling Part
    generator.add(Conv2D(64, 3, strides = 2, padding = 'same'))
    generator.add(LeakyReLU())
    generator.add(Conv2D(128, 3, strides = 2, padding = 'same'))
    generator.add(LeakyReLU())
    generator.add(Conv2D(256, 3, strides = 2, padding = 'same'))
    generator.add(LeakyReLU())
    
    # Add a bottleneck layer
    generator.add(Conv2D(256, 3, strides = 1, padding = 'same'))
    generator.add(LeakyReLU())

    # Now I need to Upsample back up
    generator.add(Conv2DTranspose(128, 3, strides = 2, padding = 'same'))
    generator.add(ReLU())
    generator.add(Conv2DTranspose(64, 3, strides = 2, padding = 'same'))
    generator.add(ReLU())
    
    # Add the output layer
    generator.add(Conv2DTranspose(3,3, padding='same', activation = 'tanh'))

    return generator


def mvp_discriminator():
    discriminator = Sequential(name = "mvp_discriminator")

    # Handle input layer
    discriminator.add(InputLayer(input_shape=(256,256, 3)))

    # Create the Downsampling Part
    discriminator.add(Conv2D(64, 3, strides = 2, padding = 'same'))
    discriminator.add(LeakyReLU())
    discriminator.add(Conv2D(128, 3, strides = 2, padding = 'same'))
    discriminator.add(LeakyReLU())
    discriminator.add(Conv2D(256, 3, strides = 2, padding = 'same'))
    discriminator.add(LeakyReLU())

    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation = 'sigmoid'))
    
    return discriminator

gen = mvp_generator()
print(gen.summary())