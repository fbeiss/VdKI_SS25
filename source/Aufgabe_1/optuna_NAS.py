import os
import json
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import re
from torchvision.datasets import ImageFolder

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
    
def prepare_data_loaders(data_dir, train_transform, val_transform, batch_size=64):
    dataset = ImageFolder(data_dir, transform=train_transform)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # Ensures full split

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Manually assign transforms (because random_split keeps transform of original dataset)
    val_data.dataset.transform = val_transform
    test_data.dataset.transform = val_transform

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f"Dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    return dataset, train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, device, patience, dataset):
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
    report = classification_report(y_true, y_pred, target_names=dataset.classes)
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
    save_results(model, dataset, history, report, conf_mat, hyperparams)

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

def optuna_objective(trial):
    conv_channels = [
        trial.suggest_int("conv1_out", 16, 500),
        trial.suggest_int("conv2_out", 32, 750),
        trial.suggest_int("conv3_out", 64, 1000),
    ]
    # Restrict kernel sizes to between 2 and 4 (input sequence length is 6)
    kernel_sizes = [
        trial.suggest_int("kernel1", 2, 4),
        trial.suggest_int("kernel2", 2, 4),
        trial.suggest_int("kernel3", 2, 4),
    ]
    fc_units_1 = trial.suggest_int("fc_units", 64, 500)
    fc_units = fc_units_1
    dropout_rate = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    epochs = 10
    patience = 5
    train_transform, val_transform = get_transforms()
    data_dir = os.path.join(base_dir, "..", "..", "data", "train")
    dataset, train_loader, val_loader, test_loader = prepare_data_loaders(data_dir, train_transform, val_transform, batch_size)
    model = SimpleCNN(num_classes=len(dataset.classes)).to(device)
    model.conv[0] = nn.Conv2d(3, conv_channels[0], kernel_size=kernel_sizes[0], padding=1)
    model.conv[3] = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=kernel_sizes[1], padding=1)
    model.conv[6] = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=kernel_sizes[2], padding=1)
    model.fc[2] = nn.Linear(conv_channels[2] * 28 * 28, fc_units)
    model.fc.append(nn.Dropout(dropout_rate))
    model.fc.append(nn.Linear(fc_units, len(dataset.classes)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopper = EarlyStopper(patience=patience)

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_loss": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
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
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}/{epochs}: Train loss={train_loss:.4f}, Train acc={train_acc:.4f}, Val acc={val_acc:.4f}, Val loss={val_loss:.4f}")

        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered")
            break

        if val_acc > best_acc:
            best_acc = val_acc

    acc, val_loss, y_true, y_pred = evaluate_model(model, val_loader, device, criterion)

    report = classification_report(y_true, y_pred, target_names=dataset.classes)
    conf_mat = confusion_matrix(y_true, y_pred)
    test_acc = test_model(model, test_loader, device)
    print("Validation Accuracy:", acc)
    


def main():
    patience = 5

    # Transforms and data
    train_t, val_t = get_transforms()
    data_dir = os.path.join(base_dir, "..", "..", "data", "train")
    dataset, train_loader, val_loader, test_loader = prepare_data_loaders(data_dir, train_t, val_t)

    # Model
    model = SimpleCNN(num_classes=len(dataset.classes)).to(device)

    while True:
        try:
            interface = CNNInterface(model, dataset.classes, device, lambda: train_model(model, train_loader, val_loader, device, patience, dataset), val_t)
            interface.launch()
        except KeyboardInterrupt:
            print("Stopping the programm.")
            break




if __name__ == "__main__":
    main()

