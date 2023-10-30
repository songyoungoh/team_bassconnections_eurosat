# Bass Connections 2023-2024
We are Bass Connections Team working on tracking climate change with satellites and artificial intelligence at Duke University. This repository provides an image classification model using PyTorch. It utilizes a ResNet-50 architecture and is trained on the EuroSAT dataset released by Helber et al. in 2019. Our evaluation metrics include a confusion matrix and accuracy rate on the test data.

## Dataset
This project utilizes the EuroSAT dataset, which offers satellite images categorized into 10 distinct classes. The dataset is available for public access and can be found on the [EuroSAT GitHub repository](https://github.com/phelber/EuroSAT). For our specific project, the data is split into a training and testing set with an 80%-20% ratio.

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

you need to download it and unzip the file. Then adjust your data path in the config.yaml

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
1. To do EDA, use 'eda.ipynb' in the 'notebooks' folder. This would give you a good sense of the EuroSAT data.
2. After doing EDA, use python files in 'src' folder to train and evaluate a model.
   * data.py: This file is to read the dataset from the given specified directory and return dataloaders along with dataset sizes.
   * model.py: This file is to import a pre-trained ResNet50 model and train it by iterating 15 epochs with a learning rate of 0.00001.
   * eval.py: This file is to calculate a test accuracy and get a confusion matrix based on the model's output.
3. Please make sure that two subfolders of the dataset should be named as 'train' and 'test' for loading train and test dataset successfully.
4. 'results' folder contains a set of models already trained by us.
5. 'reports/figures' folder includes multiple png files to show our figures.
