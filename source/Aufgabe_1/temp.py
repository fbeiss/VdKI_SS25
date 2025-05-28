import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import resnet18  # Beispielmodell
from PIL import Image
import os

base_Dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_Dir)

# 1) Modell laden (z.B. ResNet18 vortrainiert)
model = resnet18(pretrained=True)
model.eval()

# 2) Bild laden und vorverarbeiten
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open("1_rabbit_1-42_jpg.rf.22954d3ffecda0339d1b126d76624fce.jpg")  # Ersetze mit deinem Bildpfad
img_t = transform(img).unsqueeze(0)  # Batch-Dim hinzufügen

# 3) Filter der ersten Conv-Schicht visualisieren
first_conv_weights = model.conv1.weight.data.cpu()

fig, axs = plt.subplots(4, 8, figsize=(12,6))
for i, ax in enumerate(axs.flatten()):
    if i >= first_conv_weights.shape[0]:
        break
    # Filter hat Shape [3, k, k] (RGB)
    # Normalisieren für bessere Sichtbarkeit
    filt = first_conv_weights[i]
    filt_min, filt_max = filt.min(), filt.max()
    filt = (filt - filt_min) / (filt_max - filt_min)
    # Filter auf [k, k, 3] für plt.imshow
    filt = filt.permute(1, 2, 0)
    ax.imshow(filt)
    ax.axis('off')
    ax.set_title(f'Filter {i}')
plt.suptitle("Filter der ersten Conv-Schicht")
plt.show()

# 4) Aktivierungen nach erster Conv-Schicht extrahieren
def get_first_layer_activation(x):
    with torch.no_grad():
        # nur erster Layer forward
        return model.conv1(x)

activations = get_first_layer_activation(img_t).cpu().squeeze(0)  # [Channels, H, W]

fig, axs = plt.subplots(4, 8, figsize=(12,6))
for i, ax in enumerate(axs.flatten()):
    if i >= activations.shape[0]:
        break
    act = activations[i]
    # Normalisieren für Sichtbarkeit
    act_min, act_max = act.min(), act.max()
    act = (act - act_min) / (act_max - act_min)
    ax.imshow(act, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Aktivierung {i}')
plt.suptitle("Aktivierungen nach erster Conv-Schicht")
plt.show()
