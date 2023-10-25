import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import transforms, models, datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from torch.utils.data import random_split
from collections import Counter
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Custom ResNet class
class our_ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(our_ResNet, self).__init__()
        
        # Use pretrained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)
        
        # Match the number of classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = self.resnet.to(self.device)
        
        # Lists to store loss values for training and validation
        self.train_losses = []
        self.valid_losses = []

    # Forward pass
    def forward(self, x):
        return self.resnet(x)

    # Training loop
    def train_model(self, dataloaders, dataset_sizes, num_epochs=15, learning_rate=0.00001):
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.resnet.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs-1}")
            
            # Store metrics for both training and validation phases
            metrics = {
                'train': {'loss': 0.0, 'correct': 0},
                'valid': {'loss': 0.0, 'correct': 0}
            }

            # Train loop
            for phase in ['train', 'valid']:
                # As the data loader have the train and valid mode(see data.py)
                if phase == 'train':
                    self.train()
                else:
                    self.eval() 

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    
                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        # Backward only for train phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Update metrics
                    metrics[phase]['loss'] += loss.item() * inputs.size(0)
                    metrics[phase]['correct'] += torch.sum(preds == labels.data)

                # Print the log
                epoch_loss = metrics[phase]['loss'] / dataset_sizes[phase]
                epoch_acc = metrics[phase]['correct'].double() / dataset_sizes[phase]
                print(f"{phase}_loss: {epoch_loss:.4f}, {phase}_acc: {epoch_acc:.4f}")

            # For plotting
            self.train_losses.append(metrics['train']['loss'] / dataset_sizes['train'])
            self.valid_losses.append(metrics['valid']['loss'] / dataset_sizes['valid'])

    # Plotting function for training and validation losses
    def plot_losses(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Epoch vs. Train/Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Load the configuration file
    with open('../config.yaml') as p:
        config = yaml.safe_load(p)

    data_dir = config['data_dir']

    # Load datasets
    dataloaders, dataset_sizes = data_create(data_dir)
    
    # Initialize and train the model
    model1 = our_ResNet()
    model1.train_model(dataloaders, dataset_sizes, num_epochs=3)
    model1.plot_losses()

    # save the model
    torch.save(model1.state_dict(), 'trained_resnet_model.pth')

