"""Define Classifier network."""

import torch.nn as nn
from torchvision import models

class ClassificationModel(nn.Module):
    """Classifier."""

    def __init__(self,
                 input_image_size_w: int,
                 input_image_size_h: int,
                 num_classes: int,
                 backbone: str,
                 pretrained: bool = False):
        """
        Args:
            input_image_size_w/h: input image dimensions
            num_classes: number of output classes
            backbone: which backbone to use (must be pre-defined)
            pretrained: whether to use a pre-trained backbone
        """
        super().__init__()

        self.input_image_size_w = input_image_size_w
        self.input_image_size_h = input_image_size_h

        self.num_classes = num_classes

        if backbone == "r18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone == "r34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif backbone == "r50":
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unrecognized backbone")

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Run forward pass.

        Args:
            x: FloatTensor in NCHW format
        """
        logits = self.backbone(x)
        probs = self.sigmoid(logits)
        return logits, probs
