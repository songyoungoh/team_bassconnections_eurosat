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
import gdown
import zipfile
import os

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

def data_download(file_id):
    # Construct the full download URL from the file ID
    gdrive_url = f'https://drive.google.com/uc?id={file_id}'

    # Define the local file name and the download path
    zip_file = "data.zip"
    download_path = os.path.join(os.getcwd(), zip_file)

    # Download the zip file from Google Drive
    gdown.download(gdrive_url, download_path, quiet=False)

    # Create a 'data' directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Unzip the downloaded file to the 'data' directory
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall('data')

    # Remove the downloaded zip file
    os.remove(download_path)

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
