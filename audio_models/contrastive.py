import torch
import torch.nn as nn
import lightning as pl

from .encoders import Encoder1D, Encoder2D
from .losses import InfoNCELoss


class ProjectionHead(nn.Module):
    def __init__(
        self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 128
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class MultiFormatContrastiveModel(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        temperature: float = 0.07,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.encoder_1d = Encoder1D(embedding_dim=embedding_dim)
        self.encoder_2d = Encoder2D(embedding_dim=embedding_dim)

        self.projection_head_1d = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=projection_dim,
        )
        self.projection_head_2d = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=projection_dim,
        )

        self.criterion = InfoNCELoss(temperature=temperature)

    def forward(
        self, waveform: torch.Tensor, spectrogram: torch.Tensor
    ) -> tuple:
        h1 = self.encoder_1d(waveform)
        h2 = self.encoder_2d(spectrogram)

        z1 = self.projection_head_1d(h1)
        z2 = self.projection_head_2d(h2)

        return z1, z2

    def get_embeddings(
        self, waveform: torch.Tensor, spectrogram: torch.Tensor
    ) -> tuple:
        h1 = self.encoder_1d(waveform)
        h2 = self.encoder_2d(spectrogram)
        return h1, h2

    def training_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        spectrogram = batch["spectrogram"]

        z1, z2 = self(waveform, spectrogram)

        loss = self.criterion(z1, z2)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        spectrogram = batch["spectrogram"]

        z1, z2 = self(waveform, spectrogram)

        loss = self.criterion(z1, z2)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

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


class FinetuneModel(pl.LightningModule):
    def __init__(
        self,
        contrastive_model: MultiFormatContrastiveModel,
        encoder_type: str = "1d",
        num_classes: int = 10,
        freeze_encoder: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["contrastive_model"])

        self.encoder_type = encoder_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.encoder_1d = contrastive_model.encoder_1d
        self.encoder_2d = contrastive_model.encoder_2d

        if freeze_encoder:
            for param in self.encoder_1d.parameters():
                param.requires_grad = False
            for param in self.encoder_2d.parameters():
                param.requires_grad = False

        embedding_dim = (
            contrastive_model.encoder_1d.projection.out_features
            if encoder_type != "concat"
            else contrastive_model.encoder_1d.projection.out_features * 2
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        from torchmetrics import Accuracy

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(
        self, waveform: torch.Tensor, spectrogram: torch.Tensor
    ) -> torch.Tensor:
        if self.encoder_type == "1d":
            features = self.encoder_1d(waveform)
        elif self.encoder_type == "2d":
            features = self.encoder_2d(spectrogram)
        elif self.encoder_type == "concat":
            h1 = self.encoder_1d(waveform)
            h2 = self.encoder_2d(spectrogram)
            features = torch.cat([h1, h2], dim=1)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        spectrogram = batch["spectrogram"]
        y = batch["digit"]

        logits = self(waveform, spectrogram)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        spectrogram = batch["spectrogram"]
        y = batch["digit"]

        logits = self(waveform, spectrogram)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        spectrogram = batch["spectrogram"]
        y = batch["digit"]

        logits = self(waveform, spectrogram)
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
