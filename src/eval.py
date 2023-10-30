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
from data import data_create
from model import our_ResNet

# Function to evaluate the given model and return Test Accuracy.
def eval_confusion(model, dataloaders):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    dataloaders, dataset_sizes = data_create(data_dir, bs=64)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    
    # record the label
    all_labels = []
    all_predictions = []

    # get class names, as the predictions are directly numbers
    class_names = image_datasets['test'].classes
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Append labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print(f"Accuracy on test data: {100 * correct / total:.2f}%")

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Display
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized * 100, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Percentages)')
    plt.show()
    
if __name__ == "__main__":
    # Load the configuration file
    with open('../config.yaml') as p:
        config = yaml.safe_load(p)

    data_dir = config['data_dir']

    model1 = our_ResNet()

    # Load the saved model
    model1.load_state_dict(torch.load('our_resnet_model.pth'))
    eval_confusion(model1, dataloaders)

