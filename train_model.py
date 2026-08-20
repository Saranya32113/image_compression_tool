import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# -------------------------------
# PATH (NO SUBFOLDERS NEEDED)
# -------------------------------
train_dir = r"C:\Users\User\OneDrive\Desktop\image_compression_tool\dataset\train"

# -------------------------------
# SETTINGS
# -------------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 20

# -------------------------------
# LOAD IMAGES DIRECTLY
# -------------------------------
image_files = tf.data.Dataset.list_files(train_dir + "/*.png")

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, img   # autoencoder: input = output

dataset = image_files.map(load_image)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# AUTOENCODER MODEL
# -------------------------------
encoder_input = layers.Input(shape=(128, 128, 3))

x = layers.Conv2D(32, 3, activation="relu", padding="same")(encoder_input)
x = layers.MaxPooling2D(2, padding="same")(x)
x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
encoded = layers.MaxPooling2D(2, padding="same")(x)

x = layers.Conv2D(16, 3, activation="relu", padding="same")(encoded)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.UpSampling2D(2)(x)
decoded = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)

autoencoder = models.Model(encoder_input, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# -------------------------------
# TRAIN MODEL
# -------------------------------
history = autoencoder.fit(dataset, epochs=EPOCHS)

# -------------------------------
# SAVE MODEL
# -------------------------------
autoencoder.save("image_compression_model.keras")
print("✅ Training completed and model saved")

# -------------------------------
# PLOT LOSS
# -------------------------------
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
