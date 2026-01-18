from typing import Tuple
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def train_val_test_split(dataset: Dataset) -> Tuple[Subset, Subset, Subset]:
    train_idx, temp_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=np.array(dataset.labels)[temp_idx],
        random_state=42
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    return (train_ds, val_ds, test_ds)