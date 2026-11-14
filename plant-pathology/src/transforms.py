"""Transformacje obrazów: treningowe (z augmentacją) i walidacyjne/inferencyjne (bez)."""
from torchvision import transforms as T

from . import config as C


def get_train_transforms():
    return T.Compose([
        T.Resize((C.INPUT_HEIGHT, C.INPUT_WIDTH)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def get_val_transforms():
    # Bez losowości - używane w walidacji, testach i w demo
    return T.Compose([
        T.Resize((C.INPUT_HEIGHT, C.INPUT_WIDTH)),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])