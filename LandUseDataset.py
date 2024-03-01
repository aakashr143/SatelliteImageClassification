import os
import json
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from enum import StrEnum


class Mode(StrEnum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
    EXTERNAL = "external"


class LandUseDataset(Dataset):
    def __init__(self, mode: Mode = Mode.TRAIN, transform: transforms = transforms.ToTensor(), image_size: int = 256):
        self.root_dir = "./LandUseImagesDataset"
        self.mode = mode
        self.transform = transform
        self.image_size = image_size

        self.classes = {}

        self.images = []
        self.label = []

        self.__load_classes()
        self.__load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB").resize((self.image_size, self.image_size))

        if self.transform:
            img = self.transform(img)

        return img, self.label[idx]

    def __load_classes(self):
        with open(os.path.join(self.root_dir, "info.json")) as f:
            self.classes = json.load(f)

    def __load_data(self):
        for folder in os.listdir(os.path.join(self.root_dir, self.mode)):
            for img in os.listdir(os.path.join(self.root_dir, self.mode, folder)):
                self.images.append(os.path.join(self.root_dir, self.mode, folder, img))
                self.label.append(self.classes[folder])
