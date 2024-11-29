import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import matplotlib.pyplot as plt

# Load TFRecord data (assuming images are 256x256x3)
def parse_tfrecord_fn(example):
    # Define your features for TFRecord parsing
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    # Parse the example
    parsed_example = tf.io.parse_single_example(example, feature_description)
    # Decode the image from bytes to uint8
    image = tf.io.decode_jpeg(parsed_example['image'], channels=3)
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return image

# Load the TFRecord dataset
def load_tfrecord_dataset(file_pattern, batch_size=16):
    raw_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern))
    dataset = raw_dataset.map(parse_tfrecord_fn)
    dataset = dataset.batch(batch_size)
    return dataset

# Example usage: Replace 'your_dataset.tfrecord' with the actual file path
train_dataset = load_tfrecord_dataset('your_dataset.tfrecord', batch_size=8)








# Generator model (simple encoder-decoder architecture)
def build_generator():
    inputs = layers.Input(shape=(256, 256, 3))

    # Encoder: Apply a few convolutional layers with down-sampling
    x = layers.Conv2D(64, 3, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # Bottleneck: Middle layers (optional)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # Decoder: Upsampling layers to bring it back to original size
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)

    # Output layer: 3 channels (RGB)
    outputs = layers.Conv2DTranspose(3, 3, padding='same', activation='tanh')(x)

    # Define the model
    generator = Model(inputs, outputs)
    return generator






# Discriminator model (PatchGAN)
def build_discriminator():
    inputs = layers.Input(shape=(256, 256, 3))

    # Several convolutional layers
    x = layers.Conv2D(64, 3, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    # Final layer: output a single value (real or fake)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    # Define the model
    discriminator = Model(inputs, x)
    return discriminator






# Adversarial loss
def adversarial_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)

# Content loss (L2 norm between content feature maps)
def content_loss(real_content, generated_content):
    return tf.reduce_mean(tf.square(real_content - generated_content))

# Style loss (Gram matrix difference)
def gram_matrix(x):
    result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    return result / tf.cast(x.shape[1] * x.shape[2], tf.float32)

def style_loss(style_features, generated_features):
    gram_style = gram_matrix(style_features)
    gram_generated = gram_matrix(generated_features)
    return tf.reduce_mean(tf.square(gram_style - gram_generated))






# Set up the models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# Training function
def train_step(real_content_image, real_style_image):
    # 1. Train Discriminator: Real vs. Fake
    with tf.GradientTape() as disc_tape:
        generated_image = generator(real_content_image)
        
        real_output = discriminator(real_style_image)
        fake_output = discriminator(generated_image)
        
        disc_loss_real = adversarial_loss(tf.ones_like(real_output), real_output)
        disc_loss_fake = adversarial_loss(tf.zeros_like(fake_output), fake_output)
        disc_loss = (disc_loss_real + disc_loss_fake) * 0.5

    # Compute gradients and update discriminator
    grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_variables))

    # 2. Train Generator: Minimize Adversarial Loss + Content + Style Loss
    with tf.GradientTape() as gen_tape:
        generated_image = generator(real_content_image)
        
        # Get features for content and style loss (using VGG, for example)
        # In this case, we're assuming you have functions to compute content and style loss
        content_loss_value = content_loss(real_content_image, generated_image)
        style_loss_value = style_loss(real_style_image, generated_image)
        
        gen_loss = adversarial_loss(tf.ones_like(fake_output), fake_output) + content_loss_value + style_loss_value

    # Compute gradients and update generator
    grads_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(grads_gen, generator.trainable_variables))

    return disc_loss, gen_loss

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        for real_content_image, real_style_image in dataset:
            disc_loss, gen_loss = train_step(real_content_image, real_style_image)
            print(f"Epoch {epoch+1}, Discriminator Loss: {disc_loss.numpy()}, Generator Loss: {gen_loss.numpy()}")




train(train_dataset, epochs=10)
