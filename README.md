# Capstone Team Mammogram

## Introduction

The goal of our Capstone project was to explore different computer vision techniques for the classification of cancer in mammogram images of female subjects. The task was approached as a binary problem where the positive class is mammogram images with cancer and the negative class is the images without cancer. 

## Data

### Primary Dataset

The dataset used for model training and testing was obtained from the **RSNA Screening Mammography Breast Cancer Detection** Kaggle competition and consists of images from roughly 12000 patients with an average of 4 images per patient. Since the data was part of a Kaggle competition, a comprehensive test set was not available, therefore only [train_images](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data?select=train_images) was used and split into train and test sets.

The regulations for using the dataset are found in the [Rules](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/rules) section of the **RSNA Screening Mammography Breast Cancer Detection** Kaggle page. The scope of our project is exclusively educational, respecting the licenses for data use and redistribution.

### Supplemental Dataset

A supplemental dataset was used exclusevely for training in an effort to increase the number of positive examples. This dataset was also obtained from Kaggle under a page titled [DDSM-mammography-positive-case](https://www.kaggle.com/datasets/pourchot/ddsm-mammography-positive-case). The dataset was extracted from the University of South Florida’s DDSM: Digital Database for Screening Mammography and was created to be used in the development of algorithms used for classification of mammograms under the lincense CC BY 3.0 (free to be copied and redistributed in any medium or format).

## Files Description

### Data download and preprocessing

To download the data from Kaggle and perform the preprocessing steps, the user can use the notebook `data_download_preparation.ipynb`. This notebook downloads, unzips, and preprocesses the dataset, which will then be used for analysis. The important thing to note here is that to download data from Kaggle, the user needs to have an API key (in the form of .json file) downloaded in the root folder. 

### Keras Pipeline

We have some helper files plus the main notebook file. 

#### Helper files

`prepare_df.py`: It contains a class to prepare the dataframe for model training and testing. At initialization it takes a Pandas dataframe, a target variable (str), a list of colums to one-hot encode (list), the path to preprocessed images (str), the test size in percentage (float), the number of splits (int), and a seed for random state (int). Its methods are run in sequence: `.preprocess_df()` cleans the dataset from NaNs in the target variable, one-hot encodes the specified columns, and adds a column with the path to each image. `.train_test()` prepares the train/test split, stratified per target variable and grouped by patients. `.KFold()` prepares the K-Fold splits, again stratified per target variable and grouped by patients. It can also take an optional argument `balanced` (bool= in case we want to return a balanced dataset (50/50 negative/positive observations). It returns a dictionary of dictionary of dataframes, ready to be passed on to the next functions. The structure of the nested dictionaries is {fold : {'train': df_train, 'validate': df_validate, 'test': df_test}}.

`create_datasets.py`: It contains a class to perform the data feed optimization by transforming the Pandas dataframe into a Tensorflow dataset. It takes the nested dictionary as returned by our `prepare_df.KFold()` function, and it returns a dataset to be fed into our model. At initialization it takes a nested dictionary of dataframes, the fold we are interested in (int), a target variable (str), the path to images (str), a list of columns to be excluded from the metadata (list), and a double input indicator (bool). Its methods are `create_ds()`: it takes a train/validation/test indicator (str) to create the required Tensorflow dataset, and a reisizing argument in case it is needed. It maps the image path to the actual image file, resizes it, and converts it into a [0, 1] range array. The labels are also returned. In case of double input, it also reads the metadata file and includes it into the final dataset. `create_ds_autoencoder()` performs the same operations, except that it is used when we are training an autoencoder model, and thus it returns an image instead of labels. `configure_for_performance()` optimizes the dataset for training. It takes a dataset, batch size (int), train indicator (bool), and a disk cache (bool) argument. It caches the dataset (on disk if specified), shuffles it (only for the training dataset), batches it, and prefetches it.

`model_train_eval.py`: It contains our core class that controls the training. At initialization, it takes the nested dictionary, a model name (str), a path to our main directory (str), the number of splits (int), the required image size (tuple), the number of channels (int), the number of classes (int), the batch size (int), the random state (int), the number of epochs (int), a double input indicator (bool), a target variable (str), a path to images (str), and a list of columns to be excluded from metadata (list). The method `train_validate()` takes a Keras model and trains it accordingly. It returns the model training history. The method `train_validate_hyper()` performs the same, but for hyperparameters tuning. It employs another custom class to specify the hypermodel (more on this next). It returns the hypermodel tuners. 

`hypermodel.py`: It contains a custom class for hyperparameters tuning from Keras Tuner. Due to our data feeding through Tensorflow datasets, Keras Tuner is unable to consider the batch size as a hyperparameter (as batch size is controlled by the dataset itself). We therefore created a custom Hypermodel class that inherits all the properties of the standard class, but with a modified `.fit()` method, which contains the batch size hyperparameter and feeds it to the `create_datasets` class before passing it to the actual fitting function.

#### Main notebook

`model_train.ipynb`: It is our main notebook for the Keras models, and it calls all the necessary functions to do hyperparameters training, final model fitting, and evaluations for our three main "simple" models: logit, logit with double inputs, and logit with autoencoder. 

### PyTorch Pipeline

`create_splits.py`: Splits the dataset into train and validation sets where patients are unique in each set. Creates csv files with the paths and labels of images for the training and validation sets and returns training weights.

`dataset.py`: Augments images and prepares the datasets for training and validation.

`network.py`: Sets the structure for the ResNets that will be trained. 

`train.py`: Trains the classifiers, evaluates them and saves the best performing classifier as well as each classifier per epoch and returns the performance metrics of interest. 

`train_classifier.py`: Script constructed in the Google Colab environment to set the desired model configurations and train the classifier by calling the previous py files. This script is also used to produce figures 1, 2, 3 and 4 in the team's Capstone Report. To run this script simply upload it in your Google Colab environment and set its base path as the Google Drive directory where the called upon py files will be saved. Change the configurations depending on which ResNet you are planning to train and run the script. Models and metrics will be saved in the indicated Google Drive directory. 
