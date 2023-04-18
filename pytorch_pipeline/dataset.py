"""Create Classification Dataset."""

import csv
from PIL import Image
from torch.utils import data
from torchvision import transforms

class ClassificationDataset(data.Dataset):
    """Classification dataset class."""

    def __init__(
        self, csv_file: str, image_size_w: int, image_size_h: int, is_train: bool = False
    ):
        """
        Args:
            csv_file: file containing images path and label
            image_size_w/h: dimensions to resize images to
            is_train: whether to use training or testing transforms
        """
        self.csv_file = csv_file
        if is_train:
            # Use training transforms
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((image_size_h, image_size_w)),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandAugment(),
                    transforms.ToTensor(),
                ]
            )
        else:
            # Use testing transforms
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((image_size_h, image_size_w)),
                    transforms.ToTensor(),
                ]
            )

        self.img_paths = []
        self.labels = []

        # Cache labels
        self._cache_labels()

    def _cache_labels(self):
        with open(self.csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.img_paths.append(row[0])
                self.labels.append(float(row[1]))

        self.num_files = len(self.img_paths)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        label = self.labels[idx]

        # Load image
        image = Image.open(self.img_paths[idx]).convert("RGB")
        if image is None:
            raise ValueError("Fail to read {}".format(self.img_paths[idx]))

        # Apply transforms
        image = self.transforms(image)

        return {"image": image, "label": label, "name_image": self.img_paths[idx]}
