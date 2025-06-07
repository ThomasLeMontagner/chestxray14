from __future__ import annotations

import os
from typing import Optional, Callable

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ChestXray14Dataset(Dataset):
    """PyTorch Dataset for the NIH ChestX-ray14 dataset."""

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[Callable] = None,
        label_names: Optional[list[str]] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            csv_path: Path to the NIH metadata CSV file (e.g., Data_Entry_2017_v2020.csv).
            image_dir: Directory where image files are stored.
            transform: Optional transforms to apply to each image.
            label_names: Optional list of disease labels to use (if None, infer from data).
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        # Build list of unique labels if not provided
        if label_names is None:
            all_labels = self.df["Finding Labels"].apply(lambda x: x.split("|"))
            flat_labels = [label for sublist in all_labels for label in sublist]
            self.label_names = sorted(set(flat_labels))
        else:
            self.label_names = label_names

        # Multi-hot encode labels
        for label in self.label_names:
            self.df[label] = self.df["Finding Labels"].apply(
                lambda x: 1 if label in x else 0
            )

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve image and label at a specific index.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A tuple of (image_tensor, multi-hot label tensor).
        """
        row = self.df.iloc[index]
        img_path = os.path.join(self.image_dir, row["Image Index"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Create a multi-hot label tensor
        label = torch.tensor(
            [row[label] for label in self.label_names], dtype=torch.float32
        )

        return image, label
