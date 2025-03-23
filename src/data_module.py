
from __future__ import annotations

from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.dataset import ChestXray14Dataset


def create_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """Return default train and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    return train_transform, val_transform


def get_train_val_dataloaders(
    csv_path: str,
    image_dir: str,
    batch_size: int = 32,
    val_size: float = 0.1,
    label_names: Optional[list[str]] = None,
    image_size: int = 224,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Create and return train and validation dataloaders.

    Args:
        csv_path: Path to CSV metadata.
        image_dir: Directory containing image files.
        batch_size: Batch size for both loaders.
        val_size: Fraction of data to use for validation.
        label_names: Optional list of label names.
        image_size: Resize image to this size.
        num_workers: Number of subprocesses for data loading.
        seed: Random seed for reproducibility.

    Returns:
        train_loader, val_loader, list_of_labels
    """
    # Load full dataset (for consistent split)
    full_df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(full_df, test_size=val_size, random_state=seed, stratify=full_df["Finding Labels"])

    train_transform, val_transform = create_transforms(image_size)

    # Create datasets
    train_dataset = ChestXray14Dataset(
        csv_path=csv_path,
        image_dir=image_dir,
        transform=train_transform,
        label_names=label_names,
    )

    val_dataset = ChestXray14Dataset(
        csv_path=csv_path,
        image_dir=image_dir,
        transform=val_transform,
        label_names=train_dataset.label_names,  # must match
    )

    # Map indices from split DataFrames to dataset indices
    train_indices = train_df.index.tolist()
    val_indices = val_df.index.tolist()

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset.label_names
