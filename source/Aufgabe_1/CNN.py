import os
import json
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re

RESIZE = (224, 224)  # Resize images to 224x224 pixels
AUGMENTATION = {
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
NORMLIZATION = {
    "mean": [0.5131, 0.4867, 0.4275],
    "std": [0.229, 0.224, 0.225]
}
EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
base_dir = os.path.dirname(os.path.abspath(__file__))
doc_dir = os.path.join(base_dir, "..", "..", "documentation", "Aufgabe_1", "CNN", f"training_results_{current_date}")
os.makedirs(doc_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from interface_class import CNNInterface


def get_transforms():
    aug = AUGMENTATION
    norm = NORMLIZATION

    train_transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip"]),
        transforms.RandomVerticalFlip(p=aug["vertical_flip"]),
        transforms.RandomRotation(degrees=aug["rotation"]),
        transforms.ColorJitter(
            brightness=aug["color_jitter"]["brightness"],
            contrast=aug["color_jitter"]["contrast"],
            saturation=aug["color_jitter"]["saturation"],
            hue=aug["color_jitter"]["hue"]
        ),
        transforms.RandomGrayscale(p=aug["grayscale"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"])
    ])

    return train_transform, val_test_transform

class EarlyStopper:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.min_delta = delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"Early stopping counter: {self.counter}/{self.patience}")
        return self.counter >= self.patience


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
def prepare_data_loaders(data_dir, train_transform, val_transform):
    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_transform)

    classes = train_data.classes
    print(f"Classes: {classes}")
    print(f"Train data: {len(train_data)}, Val data: {len(val_data)}, Test data: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train loader: {len(train_loader)}, Val loader: {len(val_loader)}, Test loader: {len(test_loader)}")

    return classes,train_loader, val_loader, test_loader
#dsd

def train_model(model, train_loader, val_loader, device, patience, classes):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopper = EarlyStopper(patience=patience)

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_loss": []}

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_acc, val_loss,_, _ = evaluate_model(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}/{EPOCHS}: Train loss={train_loss:.4f}, Train acc={train_acc:.4f}, Val acc={val_acc:.4f}, Val loss={val_loss:.4f}")

        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered")
            break

        if val_acc > best_acc:
            best_acc = val_acc
            #torch.save(model.state_dict(), os.path.join(doc_dir, "best_model.pth"))

    acc, val_loss, y_true, y_pred = evaluate_model(model, val_loader, device, nn.CrossEntropyLoss())
    report = classification_report(y_true, y_pred, target_names=classes)
    conf_mat = confusion_matrix(y_true, y_pred)
    test_acc =  test_model(model, val_loader, device)
    print("Validation Accuracy:", acc)
    print("Validation Loss:", val_loss)
    print("Test Accuracy:", test_acc)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf_mat)


    # Plot & save
    plot_metrics(history, doc_dir, current_date)

    hyperparams = {
        "resize": RESIZE,
        "augmentation": AUGMENTATION,
        "normalization": NORMLIZATION,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE
    }
    save_results(model, history, report, conf_mat, hyperparams)

    return model, history, best_acc

def evaluate_model(model, val_loader, device, criterion):
    model.eval()
    total, correct, val_loss = 0, 0, 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()

            predicted_labels = preds.argmax(1)
            correct += (predicted_labels == yb).sum().item()
            total += yb.size(0)

            y_true.extend(yb.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())

    val_loss /= len(val_loader)
    accuracy = correct / total

    return accuracy, val_loss, y_true, y_pred

def plot_metrics(history, doc_dir, current_date):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    # Accuracy k
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(doc_dir, f"metrics_{current_date}.png"))

    plt.show()

def test_model(model, test_loader, device):
    model.eval()
    total, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(yb.cpu().tolist())
    return  correct / total



def save_results(model, history, classification, conf_matrix, hyperparams):
    results = {
        "history": history,
        "classification_report": classification,
        "confusion_matrix": conf_matrix.tolist()
    }
    data = {"hyperparameters": hyperparams, "results": results}
    with open(os.path.join(doc_dir, f"results_{current_date}.json"), 'w') as f:
        json.dump(data, f, indent=4)

    torch.save(model.state_dict(), os.path.join(doc_dir, f"final_model_{current_date}.pth"))

def find_latest_model(model_dir):
    pattern = re.compile(r"final_model_(\d{8}_\d{4})\.pth")
    latest_time = None
    latest_file = None

    for file in os.listdir(model_dir):
        match = pattern.match(file)
        if match:
            timestamp = match.group(1)
            try:
                file_time = datetime.strptime(timestamp, "%Y%m%d_%H%M")
                if latest_time is None or file_time > latest_time:
                    latest_time = file_time
                    latest_file = file
            except ValueError:
                continue

    return os.path.join(model_dir, latest_file) if latest_file else None

def load_latest_model(model_dir, device=None):
    model_path = find_latest_model(model_dir)
    if not model_path:
        raise FileNotFoundError("No valid model file found.")

    print(f"Loading model: {model_path}")
    model = SimpleCNN()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    patience = 5

    # Transforms and data
    train_t, val_t = get_transforms()
    data_dir = os.path.join(base_dir, "..", "..", "data")
    classes,train_loader, val_loader, test_loader = prepare_data_loaders(data_dir, train_t, val_t)
    print(classes)
    # Model
    model = SimpleCNN(num_classes=len(classes)).to(device)

    while True:
        try:
            interface = CNNInterface(model, classes, device, lambda: train_model(model, train_loader, val_loader, device, patience, classes), val_t)
            interface.launch()
        except KeyboardInterrupt:
            print("Stopping the programm.")
            break




if __name__ == "__main__":
    main()

