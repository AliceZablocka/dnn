"""Dataset i wagi klas (przeniesione z notebooka)."""
from typing import List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config as C


def encode_label(labels: List[str], class_list: List[str] = C.LABELS) -> torch.Tensor:
    """Lista etykiet -> wektor multi-hot."""
    target = torch.zeros(len(class_list))
    for label in labels:
        target[class_list.index(label)] = 1
    return target


class PlantDataset(Dataset):
    """Kolumna 'image' = pełna ścieżka do pliku, 'labels' = etykiety rozdzielone spacją."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['image']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, encode_label(row['labels'].split(' '))


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Odwrotność częstości klas, znormalizowana do max = 1 (jak w notebooku)."""
    all_labels = train_df['labels'].str.split(expand=True).stack()
    counts = all_labels.value_counts().reindex(C.LABELS, fill_value=1)
    weights = torch.reciprocal(torch.tensor(counts.values, dtype=torch.float))
    return weights / weights.max()