import tensorflow as tf
from tensorflow.keras import layers


def build_generator():
    # Use a ResNet-style architecture for the generator
    inputs = layers.Input(shape=(256, 256, 3))  # Assuming input image size is 256x256x3
    
    # Example: A few downsampling and residual blocks, followed by upsampling
    x = layers.Conv2D(64, kernel_size=7, strides=1, padding='same')(inputs)
    x = layers.ReLU()(x)
    
    # Residual blocks (downsample and upsample)
    for _ in range(9):  # 9 residual blocks
        res = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        res = layers.ReLU()(res)
        x = layers.Add()([x, res])
    
    # Upsample layers
    x = layers.Conv2DTranspose(3, kernel_size=7, strides=1, padding='same', activation='tanh')(x)
    
    generator = tf.keras.Model(inputs, x)
    return generator


def build_discriminator():
    # PatchGAN discriminator
    inputs = layers.Input(shape=(256, 256, 3))  # Assuming input image size is 256x256x3
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Final convolution layer that outputs one value (real or fake)
    x = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(x)
    
    discriminator = tf.keras.Model(inputs, x)
    return discriminator


def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)


def cycle_consistency_loss(real_image, cycled_image):
    return tf.reduce_mean(tf.abs(real_image - cycled_image))  # L1 loss


def identity_loss(real_image, same_image):
    return tf.reduce_mean(tf.abs(real_image - same_image))


@tf.function
def train_step(real_X, real_Y, generator_G, generator_F, discriminator_X, discriminator_Y, 
               optimizer_G, optimizer_F, optimizer_D_X, optimizer_D_Y, lambda_cycle=10, lambda_identity=0.5):
    
    with tf.GradientTape(persistent=True) as tape:
        # Generate fake images
        fake_Y = generator_G(real_X, training=True)
        fake_X = generator_F(real_Y, training=True)
        
        # Cycle consistency
        cycled_X = generator_F(fake_Y, training=True)
        cycled_Y = generator_G(fake_X, training=True)
        
        # Identity loss
        same_X = generator_F(real_X, training=True)
        same_Y = generator_G(real_Y, training=True)
        
        # Losses for the generators
        gen_G_loss = generator_loss(discriminator_Y(fake_Y)) + cycle_consistency_loss(real_X, cycled_X)
        gen_F_loss = generator_loss(discriminator_X(fake_X)) + cycle_consistency_loss(real_Y, cycled_Y)
        
        if lambda_identity > 0:
            gen_G_loss += lambda_identity * identity_loss(real_Y, same_Y)
            gen_F_loss += lambda_identity * identity_loss(real_X, same_X)

        # Discriminator losses
        disc_X_loss = discriminator_loss(discriminator_X(real_X), discriminator_X(fake_X))
        disc_Y_loss = discriminator_loss(discriminator_Y(real_Y), discriminator_Y(fake_Y))
    
    # Compute gradients and apply them
    grad_G = tape.gradient(gen_G_loss, generator_G.trainable_variables)
    grad_F = tape.gradient(gen_F_loss, generator_F.trainable_variables)
    grad_D_X = tape.gradient(disc_X_loss, discriminator_X.trainable_variables)
    grad_D_Y = tape.gradient(disc_Y_loss, discriminator_Y.trainable_variables)
    
    optimizer_G.apply_gradients(zip(grad_G, generator_G.trainable_variables))
    optimizer_F.apply_gradients(zip(grad_F, generator_F.trainable_variables))
    optimizer_D_X.apply_gradients(zip(grad_D_X, discriminator_X.trainable_variables))
    optimizer_D_Y.apply_gradients(zip(grad_D_Y, discriminator_Y.trainable_variables))
    
    return gen_G_loss, gen_F_loss, disc_X_loss, disc_Y_loss


def generate_images(generator, input_image):
    # Normalize the image to [-1, 1] for tanh activation in the generator
    input_image = input_image / 127.5 - 1
    generated_image = generator(input_image[None, ...], training=False)
    return (generated_image + 1) * 127.5  # Denormalize back to [0, 255]


# Example of training loop
for epoch in range(num_epochs):
    for real_X, real_Y in dataset:
        gen_G_loss, gen_F_loss, disc_X_loss, disc_Y_loss = train_step(real_X, real_Y, generator_G, generator_F, discriminator_X, discriminator_Y, optimizer_G, optimizer_F, optimizer_D_X, optimizer_D_Y)
    
    # Log progress (optional)
    print(f'Epoch {epoch}, G Loss: {gen_G_loss}, F Loss: {gen_F_loss}, D_X Loss: {disc_X_loss}, D_Y Loss: {disc_Y_loss}')
