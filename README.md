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
1. Download the dataset.
2. Run the notebook.
3. Follow the cells sequentially to train and evaluate the model.
