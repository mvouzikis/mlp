import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
(train_images, _), (_, _) = mnist.load_data()

# Normalize and reshape the data
train_images = train_images.astype('float32') / 255.0
train_images = np.reshape(train_images, (len(train_images), 28, 28, 1))

# Define the autoencoder model
latent_dim = 16

encoder = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(latent_dim, activation='relu')
])

decoder = models.Sequential([
    layers.Input(shape=(latent_dim,)),
    layers.Dense(7 * 7 * 8, activation='relu'),
    layers.Reshape((7, 7, 8)),
    layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')
])

autoencoder = models.Sequential([encoder, decoder])

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(train_images, train_images, epochs=10, batch_size=128, shuffle=True, validation_split=0.2)

# Test the autoencoder
encoded_images = encoder.predict(train_images)
decoded_images = decoder.predict(encoded_images)

# Display original and reconstructed images
n = 10  # Number of digits to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(train_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
