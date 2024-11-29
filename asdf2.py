import tensorflow as tf
from tensorflow.keras import layers, Sequential

def build_generator():
    # Use a ResNet-style architecture for the generator in Sequential style
    generator = Sequential(name="Generator")
    
    # Initial Conv layer
    generator.add(layers.InputLayer(input_shape=(256, 256, 3)))  # Input layer
    generator.add(layers.Conv2D(64, kernel_size=7, strides=1, padding='same'))
    generator.add(layers.ReLU())

    # Residual blocks
    for _ in range(9):  # 9 residual blocks
        res_block = Sequential()
        res_block.add(layers.Conv2D(64, kernel_size=3, strides=1, padding='same'))
        res_block.add(layers.ReLU())
        res_block.add(layers.Conv2D(64, kernel_size=3, strides=1, padding='same'))
        generator.add(layers.Add()([generator.output, res_block(generator.output)]))
    
    # Upsampling layers
    generator.add(layers.Conv2DTranspose(3, kernel_size=7, strides=1, padding='same', activation='tanh'))
    
    return generator

def build_discriminator():
    # PatchGAN discriminator in Sequential style
    discriminator = Sequential(name="Discriminator")
    
    discriminator.add(layers.InputLayer(input_shape=(256, 256, 3)))  # Input layer
    
    # Conv layers with downsampling
    discriminator.add(layers.Conv2D(64, kernel_size=4, strides=2, padding='same'))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    
    discriminator.add(layers.Conv2D(128, kernel_size=4, strides=2, padding='same'))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    
    discriminator.add(layers.Conv2D(256, kernel_size=4, strides=2, padding='same'))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    
    discriminator.add(layers.Conv2D(512, kernel_size=4, strides=2, padding='same'))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    
    # Final layer to output one value (real or fake)
    discriminator.add(layers.Conv2D(1, kernel_size=4, strides=1, padding='same'))
    
    return discriminator


d = build_discriminator()
d.summary()