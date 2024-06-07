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
from sklearn.metrics import confusion_matrix
import seaborn as sns


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

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3 , 32 , kernel_size=(3,3) , stride=1 , padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size= (3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 1)  # One output neuron for binary classification
        
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.flat(x)
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        x = self.fc4(x)
        return x

model = CNNModel()
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss with Logits
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []

num_epochs = 40

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(train_dataloader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}", colour="GREEN") as tepoch:
        for inputs, labels in tepoch:
            labels = labels.float().unsqueeze(1)  # Convert labels to float and reshape to match outputs
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Calculate training accuracy
            preds = torch.sigmoid(outputs) >= 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tepoch.set_postfix(train_loss=running_loss/len(train_dataloader), train_accuracy=correct/total)
        
        epoch_train_loss = running_loss / len(train_dataloader)
        epoch_train_accuracy = correct / total
        train_loss_history.append(epoch_train_loss)
        train_accuracy_history.append(epoch_train_accuracy)

    # Validation
    model.eval()  # Set model to evaluation mode
    val_running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(val_dataloader, unit="batch", desc="Validation", colour="RED") as tval:
            for inputs, labels in tval:
                labels = labels.float().unsqueeze(1)  # Convert labels to float and reshape to match outputs
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                preds = torch.sigmoid(outputs) >= 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                tval.set_postfix(val_loss=val_running_loss/len(val_dataloader), val_accuracy=correct/total)
    print(" "*200)
    epoch_val_loss = val_running_loss / len(val_dataloader)
    epoch_val_accuracy = correct / total
    val_loss_history.append(epoch_val_loss)
    val_accuracy_history.append(epoch_val_accuracy)


# Testing
predictions = []
true_labels = []
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        labels = labels.float().unsqueeze(1)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs) >= 0.5
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f'Accuracy of the network on the test images: {test_accuracy} %')


current_dir = os.getcwd()
images_folder = os.path.join(current_dir, 'images_CNN')
if not os.path.exists(images_folder):
    os.makedirs(images_folder)


# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(images_folder, "confusion_matrix.png"))

# Save combined metrics image
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.imshow(plt.imread(os.path.join(images_folder, "confusion_matrix.png")))
plt.axis('off')
