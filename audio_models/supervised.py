import torch
import torch.nn as nn
import lightning as pl
from torchmetrics import Accuracy

from .encoders import Encoder1D, Encoder2D


class SupervisedModel(pl.LightningModule):
    def __init__(
        self,
        encoder_type: str = "1d",
        num_classes: int = 10,
        embedding_dim: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_type = encoder_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        if encoder_type == "1d":
            self.encoder = Encoder1D(embedding_dim=embedding_dim)
        elif encoder_type == "2d":
            self.encoder = Encoder2D(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        if self.encoder_type == "1d":
            x = batch["waveform"]
        else:
            x = batch["spectrogram"]

        y = batch["digit"]

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.encoder_type == "1d":
            x = batch["waveform"]
        else:
            x = batch["spectrogram"]

        y = batch["digit"]

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        if self.encoder_type == "1d":
            x = batch["waveform"]
        else:
            x = batch["spectrogram"]

        y = batch["digit"]

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
