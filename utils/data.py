"""
Data loading utilities for supervised training.
"""

from pathlib import Path
from typing import Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(train: bool = True, use_augmentation: bool = True, augmentation_strength: str = "medium") -> transforms.Compose:
    """
    Get data transforms for training or validation.

    Args:
        train: If True, returns transforms with augmentation, else only normalization
        use_augmentation: If True, applies data augmentation (only used when train=True)
        augmentation_strength: Strength of augmentation - "light", "medium", or "strong"

    Returns:
        Composed transforms
    """
    # ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if train and use_augmentation:
        # Define augmentation strengths
        aug_params = {
            "light": {
                "crop_scale": (0.9, 1.0),
                "brightness": 0.1,
                "contrast": 0.1,
                "saturation": 0.1,
                "hue": 0.05,
                "grayscale_p": 0.0,
                "color_jitter_p": 0.5,
            },
            "medium": {
                "crop_scale": (0.8, 1.0),
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
                "grayscale_p": 0.1,
                "color_jitter_p": 0.8,
            },
            "strong": {
                "crop_scale": (0.7, 1.0),
                "brightness": 0.4,
                "contrast": 0.4,
                "saturation": 0.4,
                "hue": 0.1,
                "grayscale_p": 0.2,
                "color_jitter_p": 0.8,
            },
        }

        params = aug_params.get(augmentation_strength, aug_params["medium"])

        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=params["crop_scale"]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=params["brightness"],
                        contrast=params["contrast"],
                        saturation=params["saturation"],
                        hue=params["hue"]
                    )],
                    p=params["color_jitter_p"]
                ),
                transforms.RandomGrayscale(p=params["grayscale_p"]),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif train and not use_augmentation:
        # Training without augmentation
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        # Validation transforms
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
    use_augmentation: bool = True,
    augmentation_strength: str = "medium",
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Path to labeled data directory
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        num_workers: Number of workers for dataloaders
        seed: Random seed for reproducibility
        use_augmentation: Whether to use data augmentation on training set
        augmentation_strength: Strength of augmentation - "light", "medium", or "strong"

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
    train_dataset = datasets.ImageFolder(
        data_dir,
        transform=get_transforms(train=True, use_augmentation=use_augmentation, augmentation_strength=augmentation_strength)
    )
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
