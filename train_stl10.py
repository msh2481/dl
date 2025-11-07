import argparse
import os
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import STL10

from checkpoint_utils import load_backbone_from_checkpoint


class STL10ResNet(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 0.01, max_epochs: int = 100,
                 pretrained_backbone=None):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_backbone'])

        if pretrained_backbone is not None:
            self.model = pretrained_backbone
            num_ftrs = 512
        else:
            self.model = models.resnet18(weights=None)
            num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, 10)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class STL10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        normalize = transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713])

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(96, padding=12),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        self.val_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def setup(self, stage=None):
        self.train_dataset = STL10(self.data_dir, split="train", download=True, transform=self.train_transform)
        self.val_dataset = STL10(self.data_dir, split="test", download=True, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epoch-ratio", type=float, default=1.0)
    args = parser.parse_args()

    args.epochs = int(args.epochs * args.epoch_ratio)

    Path("checkpoints").mkdir(exist_ok=True)

    data_module = STL10DataModule(args.data_dir, args.batch_size, args.num_workers)

    pretrained_backbone = None
    if args.checkpoint:
        print(f"Loading pretrained backbone from {args.checkpoint}")
        pretrained_backbone = load_backbone_from_checkpoint(args.checkpoint)
        print("Loaded backbone successfully")

    model = STL10ResNet(args.lr, args.weight_decay, args.epochs, pretrained_backbone)

    project_name = "stl10-resnet18-finetune" if args.checkpoint else "stl10-resnet18"
    logger = WandbLogger(project=project_name, log_model=False) if "WANDB_API_KEY" in os.environ else None

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        accelerator="auto",
        devices=1,
    )

    trainer.fit(model, data_module)

    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
