import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import json
import datetime

current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
doc_dir = os.path.join(BASE_DIR, "..","..","documentation","Aufgabe_1","CNN" , f"training_results_{current_date}")
os.chdir(BASE_DIR)


# Sicherstellen, dass das Dokumentationsverzeichnis existier
if not os.path.exists(doc_dir):
    print(f"Creating documentation directory: {doc_dir}")
    os.makedirs(doc_dir)

print(f"Current working directory: {os.getcwd()}")


# Parameter für Dokumentation:
resize = (224, 224) 
augmentation = {
    "horizontal_flip": 0.5,
    "vertical_flip": 0.2,
    "rotation": 30,
    "color_jitter": {
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.3,
        "hue": 0.1
    },
    "grayscale": 0.1
}
normlization = {
    "mean": [0.5131, 0.4867, 0.4275],
    "std": [0.229, 0.224, 0.225]
}
epochs = 12
learning_rate = 1e-3
batch_size = 32


class EarlyStopper:
    def __init__(self, patience=1, delta=0):
        self.patience = patience
        self.min_delta = delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("Early stopping triggered")
                return True
        return False



# current working directory


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
train_transform = transforms.Compose([
    transforms.Resize(resize),  # Resize auf 224x224 Pixel
    transforms.RandomHorizontalFlip(p=augmentation["horizontal_flip"]),  # Horizontal flip
    transforms.RandomVerticalFlip(p=augmentation["vertical_flip"]),      # Vertikal flip (optional)
    transforms.RandomRotation(degrees=augmentation["rotation"]),         # Rotation bis ±30 Grad
    transforms.ColorJitter(
        brightness=augmentation["color_jitter"]["brightness"],
        contrast=augmentation["color_jitter"]["contrast"],
        saturation=augmentation["color_jitter"]["saturation"],
        hue=augmentation["color_jitter"]["hue"]
    ),
    transforms.RandomGrayscale(p=augmentation["grayscale"]),             # 10% Graustufen
    transforms.ToTensor(),
    transforms.Normalize(mean=normlization["mean"], std=normlization["std"])
])

val_transform = transforms.Compose([
    transforms.Resize(resize),  # Resize auf 224x224 Pixel
    transforms.ToTensor(),
    transforms.Normalize(mean=normlization["mean"], std=normlization["std"])
])


dataset = datasets.ImageFolder("../../data/train", transform=train_transform)

print(f"Dataset size: {len(dataset)} images")
print("Classes:", dataset.classes)

# Aufteilung des Datensatzes in Trainings- und Validierungsdaten (bis es Val/Test daten gibt)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

val_data.dataset.transform = val_transform  # Setze den Transform für Validierungsdaten

# Laden der Daten in batches unter Benutzung aller CPU-Threads
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
best_acc = 0
train_losses = []
train_accuracies = []

early_stopping = EarlyStopper(patience=5, delta=0.2)

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

    # Early stopping check
    if early_stopping.early_stop(epoch_loss):
        print("Early stopping triggered")
        break

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
plt.savefig(os.path.join(doc_dir, f"confusion_matrix_{current_date}.png"))
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
plt.savefig(os.path.join(doc_dir, f"train_loss_accuracy_{current_date}.png"))
plt.tight_layout()
plt.show()

print(f"Fertig. Best Validation Accuracy: {best_acc:.3f}")



os.chdir(doc_dir)


# Speichern der Hyperparameter und Modellparameter in einer JSON-Datei
hyperparams = {
    "resize": resize,
    "augmentation": augmentation,
    "normalization": normlization,
    "epochs": 15,  
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "cnn_model": str(model)
}
results = {
    "dataset_size": len(dataset),
    "best_validation_accuracy": best_acc,
    "train_losses": train_losses,
    "train_accuracies": train_accuracies,
    "classification_report": report,
    "confusion_matrix": conf_matrix.tolist(),
    "test_accuracy": accuracy,
    "test_loss": eval_loss / len(val_loader)
}


with open(f"training_results_{current_date}.json", "w") as f:
    json.dump({"hyperparameters": hyperparams, "results": results}, f, indent=4)

# Speichern der Modellparameter (Gewichte)
torch.save(model.state_dict(), f"final_model_{current_date}.pth")


