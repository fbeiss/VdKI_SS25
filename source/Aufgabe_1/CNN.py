import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

# current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# checks if cuda is available and sets the device accordingly (either gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Datenaugmentation und Normalisierung:
# Resize: Größe der Bilder auf 224x224 Pixel anpassen
# RandomHorizontalFlip: Zufälliges horizontales Spiegeln der Bilder (50% Wahrscheinlichkeit)
# RandomVerticalFlip: Zufälliges vertikales Spiegeln der Bilder (20% Wahrscheinlichkeit)
# RandomRotation: Zufällige Rotation der Bilder um ±30 Grad
# ColorJitter: Zufällige Anpassung von Helligkeit, Kontrast, Sättigung und Farbton
# RandomGrayscale: 10% der Bilder in Graustufen umwandeln
# ToTensor: Konvertiert PIL-Bilder in PyTorch-Tensoren ([0, 1] Bereich)
# Normalize: Normalisiert die Bilder mit den gegebenen Mittelwerten und Standardabweichungen
transform = transforms.Compose([
    transforms.Resize((224, 224)),                   # Resize auf 224x224 Pixel
    transforms.RandomHorizontalFlip(p=0.5),            # Horizontal flip
    transforms.RandomVerticalFlip(p=0.2),              # Vertikal flip (optional)
    transforms.RandomRotation(degrees=30),             # Rotation bis ±30 Grad
    transforms.ColorJitter(                            # Farbveränderungen
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),                 # 10% Graustufen
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5131, 0.4867, 0.4275], std=[0.229, 0.224, 0.225])
])


dataset = datasets.ImageFolder("../../data/train", transform=transform)

print(f"Dataset size: {len(dataset)} images")
print("Classes:", dataset.classes)

# Aufteilung des Datensatzes in Trainings- und Validierungsdaten (bis es Val/Test daten gibt)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

# Laden der Daten in batches unter Benutzung aller CPU-Threads
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)


# Temp CNN-Modell für die Klassifikation von Hase/Kein-Hase
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 2)  # 2 classes
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# Initialisierung des Modells und Verschiebung auf das gewählte Gerät (GPU/CPU)
model = SimpleCNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 15
best_acc = 0
train_losses = []
train_accuracies = []


for epoch in range(epochs):
    model.train()
    total, correct = 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (preds.argmax(1) == yb).sum().item()
        total += yb.size(0)
    epoch_loss = loss.item()
    epoch_accuracy = correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    train_acc = correct / total

    # Validation
    model.eval()
    val_total, val_correct = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_correct += (preds.argmax(1) == yb).sum().item()
            val_total += yb.size(0)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    # Laden des besten Modells
y_true = []
y_pred = []
eval_loss = 0.0
criterion = torch.nn.CrossEntropyLoss()  # or same as used during training

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        eval_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Step 3: Metrics
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["no rabbit", "rabbit"])

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {eval_loss / len(val_loader):.4f}")
print("Classification Report:\n", report)

# Step 4: Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["no rabbit", "rabbit"],
            yticklabels=["no rabbit", "rabbit"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 5: Plot Train Loss and Accuracy
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Train Accuracy", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Epochs")
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Fertig. Best Validation Accuracy: {best_acc:.3f}")
