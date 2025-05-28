import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Temporary transform to just load tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # Converts to [0, 1] and shape [C, H, W]
])

# Load dataset (example: folder structure)
dataset = datasets.ImageFolder("../../data/train", transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Compute mean and std
mean = 0.
std = 0.
nb_samples = 0.

for data, _ in tqdm(loader):
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)  # shape: [B, C, H*W]
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print("Mean:", mean)
print("Std:", std)
