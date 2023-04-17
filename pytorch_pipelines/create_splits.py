"""Create training and validation splits."""

# Assumes binary classification 

import os
import csv
import json
import numpy as np
import pandas as pd


def create_splits(exp_dir: str, dataset_name: str, metadata: str, train_val_split: float, train_split_name: str, val_split_name: str):
    """Create training and validation splits by sampling dataset.
    
    Args:
        exp_dir: path to experiment directory
        dataset_name: name of dataset containing images to sample
        metadata: name of dataset containing image labels
        train_val_split: ratio of data used for training (w.r.t. validation)
        train_split_name: name of training split (CSV)
        val_split_name: name of validation split (CSV)
    """
    imgs_todo = []
    imgs_done = []
    # Note: listdir() command may time out in Colab... retry until it works
    for file in os.listdir(os.path.join(exp_dir, dataset_name)):
        for img in os.listdir(os.path.join(exp_dir, dataset_name, file)):
        # Check if positive or negative
            if metadata[metadata.image_id == int(img.split('.')[0])].cancer.values == 0:
                imgs_todo.append((file, img))
            else:
                imgs_done.append((file, img))   
    

    ### Sample training / validation data ###
    
    # Make sure images from the same patient are not in both data splits
    patients = [file for file in os.listdir(os.path.join(exp_dir, dataset_name))]
    patients_train = np.random.choice(patients, size=int(len(patients) * 0.9), replace=False)
    patients_val = [patient for patient in patients if patient not in patients_train]
    
    imgs_todo_train = [img for img in imgs_todo if img[0] in patients_train]
    imgs_done_train = [img for img in imgs_done if img[0] in patients_train]
    imgs_todo_val = [img for img in imgs_todo if img[0] in patients_val]
    imgs_done_val = [img for img in imgs_done if img[0] in patients_val]
    
    print(
        f"Train TODO: {len(imgs_todo_train)} | Train DONE: {len(imgs_done_train)} | "
        f"Val TODO: {len(imgs_todo_val)} | Val DONE: {len(imgs_done_val)}"
    )

    ### Create CSV ###

    with open(os.path.join(exp_dir, train_split_name), "w") as f:
        writer = csv.writer(f)
        for img in imgs_todo_train:
            writer.writerow([os.path.join(exp_dir, dataset_name, img[0], img[1]), 0])
        for img in imgs_done_train:
            writer.writerow([os.path.join(exp_dir, dataset_name, img[0], img[1]), 1])

    with open(os.path.join(exp_dir, val_split_name), "w") as f:
        writer = csv.writer(f)
        for img in imgs_todo_val:
            writer.writerow([os.path.join(exp_dir, dataset_name, img[0], img[1]), 0])
        for img in imgs_done_val:
            writer.writerow([os.path.join(exp_dir, dataset_name, img[0], img[1]), 1])
            
    ### Return training weights ###
    
    total_imgs_train = len(imgs_todo_train) + len(imgs_done_train)
    train_weights = [total_imgs_train / len(imgs_todo_train), total_imgs_train / len(imgs_done_train)]
    return train_weights
