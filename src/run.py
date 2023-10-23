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
from eval import eval_confusion
import yaml
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(data_dir):
    # Load datasets
    dataloaders, dataset_sizes = data_create(data_dir)

    # Initialize and train the model
    model1 = our_ResNet()
    model1.train_model(dataloaders, dataset_sizes, num_epochs=3)
    model1.plot_losses()

    # Evaluate the trained model
    true_labels_str, predictions_str = eval_confusion(model1, dataloaders)

    # Optionally, you can return the results or any other information you need
    return true_labels_str, predictions_str


if __name__ == "__main__":
    # Load the configuration file
    with open('config.yaml') as p:
        config = yaml.safe_load(p)

    # Assuming the config has a key named "data_dir" that points to the path of your dataset
    data_dir = config['data_dir']

    # Run the main function
    run(data_dir)

