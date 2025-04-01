import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", "data", "images"))  # Zum Verzeichnis mit den Bildern wechseln

image = cv2.imread("yamcha.png")  # Bild laden
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV lädt in BGR, Umwandlung in RGB

plt.imshow(image)
plt.axis("off")
plt.show()

resized = cv2.resize(image, (224, 224))  # Größe auf 224x224 Pixel ändern
plt.imshow(resized)
plt.axis("off")
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Bild in Graustufen umwandeln
plt.imshow(gray, cmap="gray")
plt.axis("off")
plt.show()

blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Bild mit Gaußschem Filter glätten
plt.imshow(blurred)
plt.axis("off")
plt.show()

edges = cv2.Canny(image, 100, 200)  # Kanten mit Canny-Algorithmus erkennen
plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.show()

array_image = np.array(image)  # Bild in ein NumPy-Array umwandeln
print(array_image.shape)  # Form des Arrays ausgeben
print(array_image.dtype)  # Datentyp des Arrays ausgeben



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()  # Konvertiert in PyTorch-Tensor
])

image_pil = Image.open("yamcha.png")  # Bild mit PIL öffnen
transformed_image = transform(image_pil)
print(transformed_image.shape)  # Form des transformierten Bildes ausgeben