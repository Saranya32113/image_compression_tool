import tensorflow as tf

print("🔹 Fix model script started")

model_path = "image_compression_model.keras"

print("🔹 Trying to load model:", model_path)

model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "mse": tf.keras.losses.MeanSquaredError()
    }
)

print("✅ Model loaded successfully!")

fixed_path = "image_compression_model_FIXED.keras"
model.save(fixed_path)

print("✅ Model saved as:", fixed_path)
