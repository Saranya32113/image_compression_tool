import cv2
import numpy as np
import tensorflow as tf
import os

print("✅ Script started")
print("📁 Working directory:", os.getcwd())

# Load model (H5 ONLY)
try:
    model = tf.keras.models.load_model(
        "image_compression_model.h5",
        compile=False
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model load failed:", e)
    exit()

def compress_image(input_path, output_path="compressed_output.jpg"):
    if not os.path.exists(input_path):
        print("❌ Image not found:", input_path)
        return

    img = cv2.imread(input_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    compressed = model.predict(img)
    compressed = (compressed[0] * 255).astype(np.uint8)

    cv2.imwrite(output_path, compressed)
    print("✅ Compressed image saved as:", output_path)

if __name__ == "__main__":
    compress_image("test.jpg", "compressed_test.jpg")
