import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(root=self.root_dir, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label = os.path.basename(os.path.dirname(self.dataset.imgs[idx][0]))
        
        # Assign numerical labels based on folder names
        if label == 'fire':
            label = 1
        else:
            label = 0
        
        return img, label



transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((32, 32)),      
    transforms.Normalize(mean=[0.3,0.3,0.3], std=[1,1,1])  
])

batch_size = 64
data_path = os.path.expanduser('~/Documents/forest_fire')
dataset = CustomDataset(root_dir=data_path, transform=transform)



train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

