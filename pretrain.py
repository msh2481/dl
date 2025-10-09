"""
Pretraining script for ResNet-18 on unlabeled data using rotation prediction.

Usage:
    # Train with rotation prediction
    python pretrain.py 1e-3 --epochs 100

    # Find optimal learning rate
    python pretrain.py find
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from utils.data import get_dataloaders
from utils.lr_finder import find_lr

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


app = typer.Typer()


class RotationDataset(Dataset):
    """Dataset that applies random rotations to images for rotation prediction task."""

    def __init__(self, image_folder_dataset: datasets.ImageFolder):
        self.dataset = image_folder_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Ignore original label

        # Random rotation: 0, 90, 180, 270 degrees
        rotation = torch.randint(0, 4, (1,)).item()

        # Apply rotation
        rotated_image = torch.rot90(image, k=rotation, dims=[1, 2])

        return rotated_image, rotation


class RotationModel(nn.Module):
    """ResNet-18 backbone with rotation prediction head."""

    def __init__(self):
        super().__init__()
        # Create ResNet-18 and replace final FC with Identity
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Identity()
        self.backbone = resnet

        # Rotation prediction head (4-way classification)
        self.rotation_head = nn.Linear(512, 4)

    def forward(self, x):
        features = self.backbone(x)  # [batch_size, 512]
        rotation_logits = self.rotation_head(features)
        return rotation_logits

    def get_features(self, x):
        """Extract features from backbone (for linear probe evaluation)."""
        return self.backbone(x)


def get_unlabeled_transforms() -> transforms.Compose:
    """Get transforms for unlabeled data (no augmentation except normalization)."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_unlabeled_dataloader(
    data_dir: str = "data/train/unlabeled",
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """Create dataloader for unlabeled data with rotation task."""
    # Load images with basic transforms
    image_folder = datasets.ImageFolder(data_dir, transform=get_unlabeled_transforms())

    # Wrap with rotation dataset
    rotation_dataset = RotationDataset(image_folder)

    return DataLoader(
        rotation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def extract_features(
    model: RotationModel, dataloader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features from backbone for all samples in dataloader."""
    model.eval()
    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        features = model.get_features(images)
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def evaluate_linear_probe(
    model: RotationModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate representation quality using a linear probe on labeled data.

    Returns:
        Validation accuracy of the linear probe
    """
    logger.info("Evaluating linear probe on labeled data...")

    # Extract features
    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)

    # Train logistic regression
    classifier = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)
    classifier.fit(train_features, train_labels)

    # Evaluate
    val_predictions = classifier.predict(val_features)
    accuracy = accuracy_score(val_labels, val_predictions)

    logger.info(f"Linear probe validation accuracy: {accuracy:.4f}")
    return accuracy


def train_epoch(
    model: RotationModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
) -> tuple[float, float]:
    """Train for one epoch on rotation prediction task."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Rotation]")

    for images, rotations in pbar:
        images, rotations = images.to(device), rotations.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, rotations)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(rotations.cpu().numpy())

        # Update progress bar
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        # Log to wandb
        if use_wandb:
            wandb.log({"train/rotation_loss": loss.item(), "train/lr": current_lr})

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_targets, all_preds)

    return avg_loss, accuracy


@app.command()
def main(
    lr: str = typer.Argument(..., help="Learning rate (float) or 'find' for LR finder"),
    weight_decay: float = typer.Option(0.01, help="Weight decay for AdamW"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(128, help="Batch size for unlabeled data"),
    labeled_batch_size: int = typer.Option(
        32, help="Batch size for linear probe evaluation"
    ),
    eval_every_n_epochs: int = typer.Option(
        10, help="Evaluate linear probe every N epochs"
    ),
    num_workers: int = typer.Option(4, help="Number of dataloader workers"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Pretrain ResNet-18 using rotation prediction on unlabeled data."""
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load unlabeled data
    logger.info("Loading unlabeled data...")
    unlabeled_loader = get_unlabeled_dataloader(
        batch_size=batch_size, num_workers=num_workers
    )
    logger.info(f"Unlabeled samples: {len(unlabeled_loader.dataset)}")

    # Load labeled data for linear probe evaluation
    logger.info("Loading labeled data for evaluation...")
    labeled_train_loader, labeled_val_loader, class_names = get_dataloaders(
        batch_size=labeled_batch_size,
        val_split=0.2,
        num_workers=num_workers,
        seed=seed,
    )
    logger.info(f"Labeled train samples: {len(labeled_train_loader.dataset)}")
    logger.info(f"Labeled val samples: {len(labeled_val_loader.dataset)}")

    # Create model
    model = RotationModel().to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Setup criterion
    criterion = nn.CrossEntropyLoss()

    # LR Finder mode
    if lr == "find":
        logger.info("Running LR Finder...")
        temp_optimizer = AdamW(model.parameters(), lr=1e-7, weight_decay=weight_decay)

        find_lr(
            model=model,
            train_loader=unlabeled_loader,
            optimizer=temp_optimizer,
            criterion=criterion,
            device=device,
            start_lr=1e-7,
            end_lr=1.0,
            num_iter=200,
            output_path="lr_finder_rotation.png",
        )

        logger.info("LR finder complete. Check lr_finder_rotation.png for results.")
        return

    # Convert lr to float
    try:
        learning_rate = float(lr)
    except ValueError:
        raise ValueError(f"Invalid learning rate: {lr}. Must be a float or 'find'")

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and os.getenv("WANDB_API_KEY") is not None
    if use_wandb:
        wandb.init(
            project="rotation-pretraining",
            config={
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "architecture": "resnet18",
                "task": "rotation_prediction",
            },
        )

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Training loop
    logger.info(f"Starting pretraining for {epochs} epochs...")
    best_probe_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train on rotation task
        train_loss, train_acc = train_epoch(
            model, unlabeled_loader, optimizer, device, epoch, use_wandb
        )

        logger.info(
            f"Epoch {epoch}/{epochs} - Rotation Loss: {train_loss:.4f}, Rotation Acc: {train_acc:.4f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/rotation_epoch_loss": train_loss,
                    "train/rotation_epoch_acc": train_acc,
                }
            )

        # Evaluate linear probe periodically
        if epoch % eval_every_n_epochs == 0 or epoch == epochs:
            probe_acc = evaluate_linear_probe(
                model, labeled_train_loader, labeled_val_loader, device
            )

            if use_wandb:
                wandb.log({"eval/linear_probe_acc": probe_acc})

            # Save best model based on linear probe accuracy
            if probe_acc > best_probe_acc:
                best_probe_acc = probe_acc
                best_checkpoint_path = checkpoint_dir / "pretrained_rotation_best.pt"

                # Save only the backbone (without rotation head)
                # Create a model with 10 classes for compatibility with supervised.py
                supervised_model = models.resnet18(num_classes=10)
                supervised_model.fc = nn.Identity()

                # Copy backbone weights
                supervised_model.load_state_dict(model.backbone.state_dict())

                # Add back the fc layer
                supervised_model.fc = nn.Linear(512, 10)

                torch.save(supervised_model.state_dict(), best_checkpoint_path)
                logger.info(
                    f"Saved best checkpoint with probe_acc: {probe_acc:.4f} to {best_checkpoint_path}"
                )

        # Step scheduler
        scheduler.step()

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "pretrained_rotation_final.pt"
    supervised_model = models.resnet18(num_classes=10)
    supervised_model.fc = nn.Identity()
    supervised_model.load_state_dict(model.backbone.state_dict())
    supervised_model.fc = nn.Linear(512, 10)
    torch.save(supervised_model.state_dict(), final_checkpoint_path)

    logger.info(f"Training complete. Best probe accuracy: {best_probe_acc:.4f}")
    logger.info(f"Final checkpoint saved to {final_checkpoint_path}")

    # Cleanup
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    app()
