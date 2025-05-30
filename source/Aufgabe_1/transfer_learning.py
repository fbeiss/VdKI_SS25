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
from torchvision.models import resnet18, ResNet18_Weights
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
EPOCHS = 12
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
base_dir = os.path.dirname(os.path.abspath(__file__))
doc_dir = os.path.join(base_dir, "..", "..", "documentation", "Aufgabe_1", "transfer_learning", "resnet18" f"training_results_{current_date}")
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
        return self.counter >= self.patience

def get_transfer_model(num_classes, device): #resnet18
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    return model

def prepare_data_loaders(data_dir, train_transform, val_transform, train_split=0.8):
    dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    val_data.dataset.transform = val_transform

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    return dataset, train_loader, val_loader

def train_model(model, train_loader, val_loader, device, patience, dataset):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopper = EarlyStopper(patience=patience)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

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
        val_acc, val_loss, _, _ = evaluate_model(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{EPOCHS}: Train loss={train_loss:.4f}, Train acc={train_acc:.4f}, Val acc={val_acc:.4f}")

        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered")
            break

        if val_acc > best_acc:
            best_acc = val_acc
            #torch.save(model.state_dict(), os.path.join(doc_dir, "best_model.pth"))

            acc, val_loss, y_true, y_pred = evaluate_model(model, val_loader, device, nn.CrossEntropyLoss())
    report = classification_report(y_true, y_pred, target_names=dataset.classes)
    conf_mat = confusion_matrix(y_true, y_pred)

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
    save_results(model, dataset, history, report, conf_mat, hyperparams)

    return model, history, best_acc

def evaluate_model(model, data_loader, device, criterion=None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            if criterion:
                loss_sum += criterion(outputs, yb).item()
            preds = outputs.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(yb.cpu().tolist())

    acc = correct / total
    avg_loss = loss_sum / len(data_loader) if criterion else None
    return acc, avg_loss, all_labels, all_preds

def plot_metrics(history, doc_dir, current_date):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.grid(True)

    # Accuracy
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


def save_results(model,dataset, history, classification, conf_matrix, hyperparams):
    results = {
        "dataset_size": len(dataset),
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
    model = get_transfer_model(num_classes=2, device=device)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    patience = 5

    # Transforms and data
    train_t, val_t = get_transforms()
    data_dir = os.path.join(base_dir, "..", "..", "data", "train")
    dataset, train_loader, val_loader = prepare_data_loaders(data_dir, train_t, val_t)

    # Model
    model = get_transfer_model(len(dataset.classes), device)

    while True:
        try:
            interface = CNNInterface(model, dataset.classes, device, lambda: train_model(model, train_loader, val_loader, device, patience, dataset), val_t)
            interface.launch()
        except KeyboardInterrupt:
            print("Stopping the programm.")
            break




if __name__ == "__main__":
    main()

