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

class our_ResNet(nn.Module):
    def __init__(self, num_classes=10, learning_rate=0.00001):
        super(our_ResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = self.resnet.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.resnet.parameters(), lr=learning_rate)
        self.train_losses = []
        self.valid_losses = []

    def forward(self, x):
        return self.resnet(x)

    def train_model(self, dataloaders, dataset_sizes, num_epochs=15):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs-1}")
            metrics = {
                'train': {'loss': 0.0, 'correct': 0},
                'valid': {'loss': 0.0, 'correct': 0}
            }

            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    metrics[phase]['loss'] += loss.item() * inputs.size(0)
                    metrics[phase]['correct'] += torch.sum(preds == labels.data)

                epoch_loss = metrics[phase]['loss'] / dataset_sizes[phase]
                epoch_acc = metrics[phase]['correct'].double() / dataset_sizes[phase]
                print(f"{phase}_loss: {epoch_loss:.4f}, {phase}_acc: {epoch_acc:.4f}")

            self.train_losses.append(metrics['train']['loss'] / dataset_sizes['train'])
            self.valid_losses.append(metrics['valid']['loss'] / dataset_sizes['valid'])
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