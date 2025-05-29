import optuna
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import torch.nn as nn
import torch.optim as optim

# Simple CNN model with variable hyperparameters
class SimpleCNN(nn.Module):
    def __init__(self, trial):
        super().__init__()
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
        layers = []
        in_channels = 1
        for i in range(n_conv_layers):
            out_channels = trial.suggest_int(f'conv_{i}_out_channels', 16, 64)
            kernel_size = trial.suggest_int(f'conv_{i}_kernel_size', 3, 5)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels * 7 * 7, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Objective function for Optuna
def objective(trial):
    # Hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    epochs = 5

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(trial).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")