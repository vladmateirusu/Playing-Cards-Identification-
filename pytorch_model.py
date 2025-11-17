import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd 
import numpy as np
import sys
import os
from tqdm import tqdm

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__) 
print('Pandas version', pd.__version__)

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

data_dir = 'archive'


target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = PlayingCardDataset(data_dir, transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()

        self.base_model = timm.create_model('efficientnet_b0', pretrained=True) 
        self.features = nn.Sequential(*list(self.base_model.children())[:-1]) 
        enet_out_size = 1280 
        self.classifier = nn.Linear(enet_out_size, num_classes) 
        pass

    def forward(self, x): 
        x = self.features(x)
        output = self.classifier(x)
        return output

model = SimpleCardClassifier(num_classes=53)

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimize
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_folder = 'archive/train'
valid_folder = 'archive/valid'
test_folder = 'archive/test'

train_dataset = PlayingCardDataset(train_folder, transform=transform)
valid_dataset = PlayingCardDataset(valid_folder, transform=transform)
test_dataset = PlayingCardDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # keep the training shuffled
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_dataset_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_epoch = 5 # epoch is one run through entire training set
train_losses, val_losses = [], []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epoch):
    # Set the model to train
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # back propagation which will update model weight in every step of the way
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc='Validation loop'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(valid_loader.dataset)
    val_losses.append(val_losses)

    print(f"Epoch {epoch+1}/{num_epoch} - Train loss: {train_loss}, Validation loss: {val_loss}")