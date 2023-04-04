# Capstone Team Mammogram

## Introduction

The goal of our Capstone project was to explore different computer vision techniques for the classification of cancer in radiographic breast images of female subjects. The task was approached as a binary problem where the positive class is radiographic breast images with cancer and the negative class is the images without cancer. 

## Data


The dataset used for model training and testing was obtained from the **RSNA Screening Mammography Breast Cancer Detection** Kaggle competition and consists of images from roughly 12000 patients with an average of 4 images per patient. Since the data was part of a Kaggle competition, a comprehensive test set was not available, therefore only [train_images](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data?select=train_images) was used and split into train and test sets.


The regulations for using the dataset are found in the [Rules](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/rules) section of the **RSNA Screening Mammography Breast Cancer Detection** Kaggle page. The scope of our project is exclusively educational, respecting the licenses for data use and redistribution.


## Files Description

`create_splits.py`: Splits the dataset into train and validation sets where patients are unique in each set. Creates csv files with the paths and labels of images for the training and validation sets and returns training weights.


`dataset.py`: Augments images and prepares the datasets for training and validation.


`network.py`: Sets the structure for the CNNs that will be trained. 


`train.py`: Trains the classifiers, evaluates them and returns the best performing classifier and the metrics of interest. 


`treain_classifier.py`: Script set in the Google Collab environment to set the desired model configs and train the classifier by calling the previous py files. 
