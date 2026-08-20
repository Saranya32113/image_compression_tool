import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("image_compression_model.h5")

img_path = "test.jpg"
  # Change this to any image

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (128,128)) / 255.0
input_img = img_resized.reshape(1,128,128,3)

output = model.predict(input_img)

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img)

plt.subplot(1,2,2)
plt.title("Compressed Output")
plt.imshow(output[0])
plt.show()
