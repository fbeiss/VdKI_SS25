import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os


# current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# checks if cuda is available and sets the device accordingly (either gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Datenaugmentation und Normalisierung:
# Resize: Größe der Bilder auf 224x224 Pixel anpassen
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
    transforms.Normalize(mean=[0.5527, 0.5293, 0.4742], std=[0.2510, 0.2463, 0.2474])
])

print("Transformations:", transform)

dataset = datasets.ImageFolder("../../data/train", transform=transform)

print(f"Dataset size: {len(dataset)} images")
print("Classes:", dataset.classes)
print("Class indices:", dataset.class_to_idx)
print("Sample image shape:", dataset[0][0].shape)  # Shape of the first image tensor
print("Sample label:", dataset[0][1])  # Label of the first image


train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32)


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

model = SimpleCNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 10
best_acc = 0


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

print(f"Fertig. Best Validation Accuracy: {best_acc:.3f}")
