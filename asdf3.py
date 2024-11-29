import tensorflow as tf
from tensorflow.keras import layers, Sequential


def mvp_generator():
    genny = Sequential(name="Genny")

    genny.add(layers.InputLayer(input_shape=(256,256, 3)))
    genny.add(layers.Conv2D(64, kernel_size = 7, strides = 1, padding = 'same'))
    genny.add(layers.ReLU())
    
    x = layers.Conv2D(64, kernel_size=7, strides = 1, padding = 'same')(inputs)

def mvp_discriminator():
    discr = Sequential(name = "Discr")

    discr.add(layers.InputLayer(input_shape=(256,256,3)))
    discr.add(layers.Conv2D(64, kernel_size = 4, strides = 2, padding = 'same'))



    # Final layer to output one value (real or fake)
    discr.add(layers.Conv2D(1, kernel_size=4, strides=1, padding='same'))