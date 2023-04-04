"""Training script."""

import numpy as np
import os
import shutil
from sklearn import metrics
import time
from tqdm import tqdm
from typing import Any, Dict, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

import dataset
import network

class Trainer:
    """Classification model Trainer."""

    def __init__(self,
                 train_configs: Dict[str, Any],
                 model_configs: Dict[str, Any],
                 exp_dir: str,
                 train_weights: Optional[List[float]] = None,
                 desc: str = ''
    ):
        """
        Args:
            train_configs: dict of training configs
            model_configs: dict of model configs
            exp_dir: path to experiment directory
            train_weights: training weights (use if dataset is unbalanced)
            desc: training description, will be added to training folder name
        """
        torch.manual_seed(42)

        # Parse training config
        self._parse_configs(train_configs)

        # Load model
        self.model = network.ClassificationModel(**model_configs)
        self.model.to(device=self.device)

        # Load a pre-trained checkpoint if provided
        if self.starting_ckpt is not None:
            ckpt = torch.load(self.starting_ckpt)
            self.model.load_state_dict(ckpt["model"])

        # Set optimizer, scheduler and criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        self.train_weights = torch.FloatTensor(train_weights).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.train_weights)

        # Load data
        self._load_data()

        # Create output directory
        dirname_xp = f"{int(time.time())}"
        if desc is not None:
            dirname_xp += f"_{desc}"
        self.outdir = os.path.join(exp_dir, dirname_xp)
        os.makedirs(self.outdir)

        # Copy config files to output directory
        #shutil.copyfile(train_configs, os.path.join(self.outdir, "train_configs.yml"))
        #shutil.copyfile(model_configs, os.path.join(self.outdir, "model_configs.yml"))

        # Tensorboard
        self.tb_writer = SummaryWriter(log_dir=self.outdir)
        #%tensorboard --logdir self.outdir

        # Paths to save best and last model
        self.path_best_model = os.path.join(self.outdir, "model_best.pt")
        self.path_last_model = os.path.join(self.outdir, "model_last.pt")

    def _parse_configs(self, configs: Dict[str, Any]):
        """Parse configs."""
        for main_key in configs.keys():
            for key in configs[main_key].keys():
                setattr(self, key, configs[main_key][key])

    def _load_data(self):
        """Load training and validation datasets."""
        train_dataset = dataset.ClassificationDataset(
            self.training_split,
            self.model.input_image_size_w,
            self.model.input_image_size_h,
            is_train=True
        )
        val_dataset = dataset.ClassificationDataset(
            self.validation_split,
            self.model.input_image_size_w,
            self.model.input_image_size_h,
            is_train=False,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

    def train_epoch(self, epoch: int):
        """Train model for 1 epoch."""
        self.model.train()

        n_batches = len(self.train_loader)

        loss_epoch = 0.0
        accuracy_epoch = 0.0
        for i, batch in tqdm(enumerate(self.train_loader), total=n_batches, desc="Training"):
            images = batch["image"].to(device=self.device)
            labels = batch["label"].to(device=self.device)

            self.optimizer.zero_grad()

            logits, _ = self.model(images)
            logits = torch.squeeze(logits)

            # Loss and optimizer step
            loss = self.criterion(logits, labels.float()) #fix error
            loss.backward()
            self.optimizer.step()

            # Get predictions (using argmax i.e. assuming 0.5 threshold)
            preds = torch.argmax(logits, dim=1)

            loss_epoch += loss.item()
            accuracy = torch.sum(preds == labels) / len(labels)
            accuracy_epoch += accuracy

            # Tensorboard bookkeeping
            self.tb_writer.add_scalar("train/loss", loss, epoch * n_batches + i)
            self.tb_writer.add_scalar("train/accuracy", accuracy, epoch * n_batches + i)

        mean_loss = loss_epoch / n_batches
        mean_accuracy = accuracy_epoch / n_batches
        # Tensorboard bookkeeping
        self.tb_writer.add_scalar("train/mean_loss", mean_loss, epoch)
        self.tb_writer.add_scalar("train/mean_accuracy", mean_accuracy, epoch)

        return mean_loss, mean_accuracy

    def evaluate(self, epoch: int):
        """Evaluate model --> compute accuracy, precision, recall and F1-score."""
        self.model.eval()

        n_batches = len(self.val_loader)

        loss_sum = 0
        all_labels = []
        all_preds = []
        categories = ["todo", "done"]
        for i, batch in tqdm(enumerate(self.val_loader), total=n_batches, desc="Validation"):
            images = batch["image"].to(device=self.device)
            labels = batch["label"].to(device=self.device)

            with torch.no_grad():
                logits, _ = self.model(images)
                logits = torch.squeeze(logits)
                loss = self.criterion(logits, labels)
                # Get predictions (using threshold 0 for logits i.e. 0.5 for probabilities)
                preds = torch.argmax(logits, dim=1).cpu()

            all_labels += labels.tolist()
            all_preds += preds.tolist()

            loss_sum += loss.item()

            # Tensorboard bookkeeping
            self.tb_writer.add_scalar("val/loss", loss, epoch * n_batches + i)

        cls_report = metrics.classification_report(
            all_labels, all_preds, target_names=categories, zero_division=0
        )

        print(cls_report)

        prec, rec, f1, _ = metrics.precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=[0, 1], zero_division=0
        )

        accuracy = (np.array(all_labels) == np.array(all_preds)).sum() / len(all_labels)

        mean_loss = loss_sum / n_batches
        # Tensorboard bookkeeping
        self.tb_writer.add_scalar("val/mean_loss", mean_loss, epoch)
        self.tb_writer.add_scalar("val/accuracy", accuracy, epoch)
        self.tb_writer.add_scalar("val/precision", prec[1], epoch)
        self.tb_writer.add_scalar("val/recall", rec[1], epoch)
        self.tb_writer.add_scalar("val/f1", f1[1], epoch)

        return loss, accuracy, prec, rec, f1

    def train(self):
        """
        Train model on given number of epochs, record training and validation metrics,
        and save best model checkpoint based on validation accuracy.
        """
        # Print experiment folder
        print(f"\n===== Experiment folder: {self.outdir} =====")

        # Create results file to store training and validation metrics
        self.results_file = os.path.join(self.outdir, "all_metrics.csv")
        # Write header
        with open(self.results_file, "w") as f:
            f.write("Epoch, Train Loss, Val Loss, Train Acc, Val Acc, Val Precision, Val Recall, Val F1")

        best_perf = 0
        best_loss = float("inf")
        best_epoch = 0
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1} / {self.num_epochs}")

            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # Evaluate on validation set
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.evaluate(epoch)

            # Print metrics
            print(
                f"Train Loss: {train_loss} | Val Loss: {val_loss} | Train Acc: {train_acc} |"
                f" Val Acc: {val_acc}"
            )

            # Write metrics to results file
            with open(self.results_file, "a") as f:
                f.write(
                    f"\n{epoch + 1}, {train_loss}, {val_loss}, {train_acc}, {val_acc}, {val_prec[1]},"
                    f" {val_rec[1]}, {val_f1[1]}"
                )

            ckpt = {"model": self.model.state_dict()}

            if (val_acc > best_perf or (val_acc == best_perf and val_loss <= best_loss)):
              # change this cause val_loss can be 0 at epoch 1...
                best_perf = val_acc
                best_loss = val_loss
                best_epoch = epoch + 1

                # Save new best model
                torch.save(ckpt, self.path_best_model)
                print(">>> New best model saved!")

            # Save last model
            torch.save(ckpt, self.path_last_model)

        self.scheduler.step()

        # Print experiment folder again at the end
        print(f"\n>>> Training done. Best epoch: {best_epoch}. Val mAP: {best_perf}.")
        print(f"\n===== Experiment folder: {self.outdir} =====")
