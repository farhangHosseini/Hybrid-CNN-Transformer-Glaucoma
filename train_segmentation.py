import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from models.unet_model import UNet
from utils.datasets import SegmentationDataset, get_transforms

# Configuration
EPOCHS = 25
BATCH_SIZE = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # Placeholder for paths - User needs to update these
    image_paths = [] # ["data/images/img1.jpg", ...]
    mask_paths = []  # ["data/masks/mask1.png", ...]
    
    if not image_paths:
        print("Please configure dataset paths in script.")
        return

    train_ds = SegmentationDataset(image_paths, mask_paths, transform=get_transforms('train'))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train()