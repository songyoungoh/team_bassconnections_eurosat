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

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def data_create(data_dir):
  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
  dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'test']}
  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
  train_dataset = image_datasets['train']
  train_size = int(0.8 * len(train_dataset))
  valid_size = len(train_dataset) - train_size
  train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])

  dataloaders['train'] = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
  dataloaders['valid'] = torch.utils.data.DataLoader(valid_subset, batch_size=32, shuffle=False, num_workers=4)
  dataset_sizes['train'] = len(train_subset)
  dataset_sizes['valid'] = len(valid_subset)
  return dataloaders, dataset_sizes