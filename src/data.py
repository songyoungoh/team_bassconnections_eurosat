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

device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")

# reshape the img to the desired size, and do the normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # this filter is normally used in rgb img, https://pytorch.org/vision/stable/models.html
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Function to read the dataset from the specified directory and return dataloaders and dataset sizes.
def data_create(data_dir, bs=64):
    # Create a dictionary of ImageFolder datasets for both training and testing sets.
    
    # ImageFolder expects data loader as:
    #     data_dir/train/class1/xxx.png
    #     data_dir/train/class2/xxx.png
    #     ...
    #     data_dir/test/class1/xxx.png
    #     ...
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    
    # Create dataloaders for the datasets with a batch size of bs. 
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs, shuffle=True) for x in ['train', 'test']}
    
    # Get the size of each dataset
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    
    # Split the training dataset into a training set and a validation set. 
    train_dataset = image_datasets['train']
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])
    
    dataloaders['train'] = torch.utils.data.DataLoader(train_subset, batch_size=bs, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(valid_subset, batch_size=bs)
    
    # Update the dataset sizes for training and validation subsets.
    dataset_sizes['train'] = len(train_subset)
    dataset_sizes['valid'] = len(valid_subset)
    
    # Return the dataloaders and dataset sizes.
    return data loaders, dataset_sizes
