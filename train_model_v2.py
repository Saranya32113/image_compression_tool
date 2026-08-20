from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# -------------------------------
# Paths
# -------------------------------
train_dir = r"dataset/train"
val_dir   = r"dataset/val"

# -------------------------------
# Settings
# -------------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# -------------------------------
# Function to add Gaussian noise
# -------------------------------
def add_noise(img):
    noise = np.random.normal(0, 0.05, img.shape)  # mean=0, std=0.05
    img_noisy = np.clip(img + noise, 0., 1.)
    return img_noisy

# -------------------------------
# Data generators with augmentation + noise
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    preprocessing_function=add_noise  # <-- apply Gaussian noise
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="input"
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="input"
)

# -------------------------------
# Build autoencoder
# -------------------------------
input_img = layers.Input(shape=(128,128,3))

# Encoder
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2,2), padding='same')(x)

# Decoder
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)

# -------------------------------
# Compile model
# -------------------------------
autoencoder.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')

# -------------------------------
# Train model
# -------------------------------
history = autoencoder.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -------------------------------
# Save model
# -------------------------------
autoencoder.savemodel = tf.keras.models.load_model("image_compression_model.keras", compile=False)
print("Training done! Saved as image_compression_model.keras")
