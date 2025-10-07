"""
Data loading utilities for supervised training.
"""

from pathlib import Path
from typing import Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Get data transforms for training or validation.

    Args:
        train: If True, returns transforms with augmentation, else only normalization

    Returns:
        Composed transforms
    """
    # ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if train:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )


def get_dataloaders(
    data_dir: str = "data/train/labeled",
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Path to labeled data directory
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        num_workers: Number of workers for dataloaders
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Load full dataset with train transforms (we'll apply val transforms separately)
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes

    # Get indices for train/val split (stratified)
    targets = [full_dataset.targets[i] for i in range(len(full_dataset))]
    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=val_split,
        stratify=targets,
        random_state=seed,
    )

    # Create separate datasets with different transforms
    train_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(train=True))
    val_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(train=False))

    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names
