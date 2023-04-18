"""Create training and validation splits."""

# Assumes binary classification 

import os
import csv
import json
import numpy as np
import pandas as pd
import random


def create_splits(exp_dir: str, dataset_name: str, supplemental_pos: str, metadata: str, train_val_split: float, train_split_name: str, val_split_name: str):
    """Create training and validation splits by sampling dataset.
    
    Args:
        exp_dir: path to experiment directory
        dataset_name: name of dataset containing images to sample
        metadata: name of dataset containing image labels
        train_val_split: ratio of data used for training (w.r.t. validation)
        train_split_name: name of training split (CSV)
        val_split_name: name of validation split (CSV)
    """
    imgs_negative = []
    imgs_positive = []
    # Note: listdir() command may time out in Colab... retry until it works
    for file in os.listdir(os.path.join(exp_dir, dataset_name)):
        for img in os.listdir(os.path.join(exp_dir, dataset_name, file)):
        # Check if positive or negative
            if metadata[metadata.image_id == int(img.split('.')[0])].cancer.values == 0:
                imgs_negative.append((file, img))
            else:
                imgs_positive.append((file, img))   
                
                
    # Getting supplemental positive images for training 
    supplemental_list = []
    for file in os.listdir(os.path.join(exp_dir, supplemental_pos)):
        for img in os.listdir(os.path.join(exp_dir, supplemental_pos, file)):
            supplemental_list.append((file, img))  

    ### Sample training / validation data ###
    
    # Make sure images from the same patient are not in both data splits
    patients = [file for file in os.listdir(os.path.join(exp_dir, dataset_name))]
    patients_train = random.sample(patients, int(len(patients) * 0.9))
    patients_val = [patient for patient in patients if patient not in patients_train]
    
    # Downsample the negative class - training
    negative_train = [img for img in imgs_negative if img[0] in patients_train]
    imgs_negative_train = random.sample(negative_train, int(len(negative_train) * 0.1))
    imgs_positive_train = [img for img in imgs_positive if img[0] in patients_train]
    # Downsample the negative class - validation
    negative_val = [img for img in imgs_negative if img[0] in patients_val] 
    imgs_negative_val = random.sample(negative_val, int(len(negative_val) * 0.1))
    imgs_positive_val = [img for img in imgs_positive if img[0] in patients_val]
    
    len_train_pos = len(imgs_positive_train) + len(supplemental_list)
    
    print(
        f"Train Negative: {len(imgs_negative_train)} | Train Positive: {len_train_pos} | "
        f"Val Negative: {len(imgs_negative_val)} | Val Positive: {len(imgs_positive_val)}"
    )

    ### Create CSV ###

    with open(os.path.join(exp_dir, train_split_name), "w") as f:
        writer = csv.writer(f)
        for img in imgs_negative_train:
            writer.writerow([os.path.join(exp_dir, dataset_name, img[0], img[1]), 0])
        for img in imgs_positive_train:
            writer.writerow([os.path.join(exp_dir, dataset_name, img[0], img[1]), 1])
        for img in supplemental_list:
            writer.writerow([os.path.join(exp_dir, supplemental_pos, img[0], img[1]), 1])

    with open(os.path.join(exp_dir, val_split_name), "w") as f:
        writer = csv.writer(f)
        for img in imgs_negative_val:
            writer.writerow([os.path.join(exp_dir, dataset_name, img[0], img[1]), 0])
        for img in imgs_positive_val:
            writer.writerow([os.path.join(exp_dir, dataset_name, img[0], img[1]), 1])
            
    ### Return training weights ###
    
    total_imgs_train = len(imgs_negative_train) + len(imgs_positive_train)
    train_weights = [total_imgs_train / len(imgs_negative_train), total_imgs_train / len(imgs_positive_train)]
    return train_weights
