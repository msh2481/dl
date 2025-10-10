"""
Generate submission file for Kaggle.

Usage:
    python submit.py checkpoints/best.pt --output submission.csv
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import typer
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

app = typer.Typer()


class TestDataset(Dataset):
    """Dataset for test images."""

    def __init__(self, test_dir: str, transform):
        self.test_dir = Path(test_dir)
        self.image_files = sorted(
            self.test_dir.glob("*.jpg"), key=lambda x: int(x.stem)
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, image_path.name


def get_test_transforms() -> transforms.Compose:
    """Get transforms for test data."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@app.command()
def main(
    checkpoint: str = typer.Argument(..., help="Path to trained model checkpoint"),
    output: str = typer.Option("submission.csv", help="Output CSV file path"),
    test_dir: str = typer.Option("data/test", help="Test data directory"),
    batch_size: int = typer.Option(64, help="Batch size for inference"),
    num_workers: int = typer.Option(4, help="Number of dataloader workers"),
):
    """Generate submission file for Kaggle."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {checkpoint}")
    model = models.resnet18(num_classes=10)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Class names (sorted alphabetically to match ImageFolder)
    class_names = [
        "butterfly",
        "cat",
        "chicken",
        "cow",
        "dog",
        "elephant",
        "horse",
        "sheep",
        "spider",
        "squirrel",
    ]

    # Create test dataset
    logger.info(f"Loading test data from {test_dir}")
    test_dataset = TestDataset(test_dir, transform=get_test_transforms())
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = []

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            for filename, pred in zip(filenames, preds):
                predictions.append({"id": filename, "class": class_names[pred]})

    # Create DataFrame and save
    df = pd.DataFrame(predictions)
    df.to_csv(output, index=False)
    logger.info(f"Submission saved to {output}")
    logger.info(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    app()
