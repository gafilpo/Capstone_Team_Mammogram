"""train_classifier.ipynb
Automatically generated by Colaboratory.
# Initialization
"""

# Mount drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

# Commented out IPython magic to ensure Python compatibility.
# Set up base path
import os
import numpy as np
import pandas as pd

BASE_PATH = "/content/gdrive/MyDrive/Capstone/Project"

# %cd $BASE_PATH

"""# Configs"""

# Configs to change

EXP_DIR = "" # experiment directory
DATASET_NAME = "images_resized"  # name of (zip) folder containing your data
NUM_CLASSES = 1 # number of classes
METADATA = pd.read_csv("metadata.csv")

# Fixed configs (change for experimentations only)

# Datasets configs
TRAIN_SPLIT_NAME = "train.csv"
VAL_SPLIT_NAME = "val.csv"
TRAIN_VAL_SPLIT = 0.9

# Model configs
INPUT_IMAGE_SIZE_W = 1000
INPUT_IMAGE_SIZE_H = 1000
BACKBONE = "r18"

MODEL_CONFIGS = {
    "input_image_size_w": INPUT_IMAGE_SIZE_W,
    "input_image_size_h": INPUT_IMAGE_SIZE_H,
    "num_classes": NUM_CLASSES,
    "backbone": BACKBONE,
    "pretrained": True
}

# Training configs
DESC = "training" # experiment description
TRAIN_CONFIGS = {
    "dataset": {
        "training_split": os.path.join(BASE_PATH, EXP_DIR, TRAIN_SPLIT_NAME),
        "validation_split": os.path.join(BASE_PATH, EXP_DIR, VAL_SPLIT_NAME),
    },

    "training_parameters": {
        "num_epochs": 15,
        "batch_size": 8,
        "base_lr": 0.001,
        "device": 'cuda:0',
        "starting_ckpt": None,
    }
}

# Note: configs like optimizer, criterion and schedulers are set inside Trainer class

"""# Training"""

# Unzip dataset
#!unzip "{BASE_PATH}{EXP_DIR}/{DATASET_NAME}.zip" -d "{BASE_PATH}/{EXP_DIR}"

# Check number of images is correct (sometimes Colab doesn't unzip correctly)
!ls {DATASET_NAME} |wc -l

import create_splits
import network
import train


# Create training and validation splits and return training weights (negative / positive ratio)
train_weights = create_splits.create_splits(
    exp_dir = BASE_PATH,
    dataset_name = DATASET_NAME,
    metadata = METADATA,
    train_val_split = TRAIN_VAL_SPLIT,
    train_split_name = TRAIN_SPLIT_NAME,
    val_split_name = VAL_SPLIT_NAME,
)

# Train

trainer = train.Trainer(model_configs=MODEL_CONFIGS,
                        train_configs=TRAIN_CONFIGS,
                        exp_dir=os.path.join(BASE_PATH, EXP_DIR),
                        train_weights=train_weights,
                        desc=DESC)

trainer.train()

"""# Tensorboard (Optional)"""

# Kill existing sessions
#!kill $(ps -e | grep 'tensorboard' | awk '{print $1}')

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# %tensorboard --logdir "{BASE_PATH}/{EXP_DIR}/{TRAIN_EXP}"
