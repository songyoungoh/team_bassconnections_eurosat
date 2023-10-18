# Bass Connections 2023-2024
We are Bass Connections Team working on tracking climate change with satellites and artificial intelligence at Duke University. This repository provides an image classification model using PyTorch. It utilizes a ResNet-50 architecture and is trained on EuroSAT dataset which was released by Helber et al. in 2019. Our evaluation metrics include a confusion matrix and accuracy rate on the test data.

## Dataset
The dataset used in this project is sourced from the EuroSAT (https://github.com/phelber/EuroSAT) comprised of 10 different classes. We use a 80%-20% of training-test split ratio. You can download it via https://drive.google.com/file/d/158N0Rg0tjCBMDtJrmz8tPPIy-u9OprGR/view?usp=drive_link

## Requirements
- **torch**
- **torchvision**
- **sklearn**
- **numpy**
- **pandas**
- **matplotlib**
- **seaborn**
- **os**
- **random**
- **collections**
- **ssl**

## Usage
1. To do EDA, use 'eda.ipynb' which would give you a good sense of the EuroSAT data.
2. After doing EDA, use 'main.ipynb' to train and evaluate a model. We encourage you to open this file on Google Colab.
3. Before running 'main.ipynb', please make sure to download the dataset in your google drive('/content/drive/MyDrive/Train_Test_Splits') using 'Train_Test_Splits' as a folder name.
4. Please also make sure that two subfolders should be named as 'train' and 'test' for loading train and test dataset respectively.
5. Follow the cells in 'main.ipynb' sequentially. This file already includes code for unzipping data for you.
