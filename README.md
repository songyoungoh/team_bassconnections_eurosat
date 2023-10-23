1. create src/data/result/config.yaml
2. for src, separate main.ipynb into data.py/train.py/eva.py
3. save the fig into the result folder.


# Bass Connections 2023-2024
We are Bass Connections Team working on tracking climate change with satellites and artificial intelligence at Duke University. This repository provides an image classification model using PyTorch. It utilizes a ResNet-50 architecture and is trained on the EuroSAT dataset released by Helber et al. in 2019. Our evaluation metrics include a confusion matrix and accuracy rate on the test data.

## Dataset
This project utilizes the EuroSAT dataset, which offers satellite images categorized into 10 distinct classes. The dataset is available for public access and can be found on the [EuroSAT GitHub repository](https://github.com/phelber/EuroSAT). You can download it via https://drive.google.com/file/d/158N0Rg0tjCBMDtJrmz8tPPIy-u9OprGR/view?usp=drive_link. For our specific project, the data is split into a training and testing set with an 80%-20% ratio.

The dataset is organized in the following directory hierarchy:

```
Train_Test_Splits
│
├── train
│   ├── AnnualCrop
│   ├── [Other Classes]
│
└── test
    ├── AnnualCrop
    ├── [Other Classes]
```

## Requirements
matplotlib==3.7.2
numpy==1.23.5
pandas==2.0.3
protobuf==4.23.4
scikit_learn==1.3.0
seaborn==0.13.0
torch==2.0.1
torchvision==0.15.2

use pip to install all the requirements.
```
pip install -r requirements.txt
```

## Usage
1. To do EDA, use 'eda.ipynb' which would give you a good sense of the EuroSAT data.
2. After doing EDA, use 'main.ipynb' to train and evaluate a model. We encourage you to open this file on Google Colab.
3. Before running 'main.ipynb', please make sure to download the dataset in your google drive('/content/drive/MyDrive/Train_Test_Splits') using 'Train_Test_Splits' as a folder name.
4. Please also make sure that two subfolders should be named as 'train' and 'test' for loading train and test dataset respectively.
5. Follow the cells in 'main.ipynb' sequentially. This file already includes code for unzipping data for you.
