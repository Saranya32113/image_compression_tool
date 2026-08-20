import matplotlib.pyplot as plt
import cv2

img = cv2.imread("dataset/train/cars/your_image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis("off")
plt.show()
