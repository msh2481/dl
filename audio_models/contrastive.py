"""Multi-format contrastive learning model."""

import torch
import torch.nn as nn
import lightning as pl

from .encoders import Encoder1D, Encoder2D
from .losses import InfoNCELoss


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Args:
        input_dim: Input dimension (default: 512)
        hidden_dim: Hidden layer dimension (default: 512)
        output_dim: Output dimension (default: 256)
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input embeddings (batch, input_dim)

        Returns:
            Projected embeddings (batch, output_dim)
        """
        return self.projection(x)


class MultiFormatContrastiveModel(pl.LightningModule):
    """Multi-format contrastive learning model.

    Contrasts embeddings from raw waveforms (1D encoder) and spectrograms (2D encoder).
    Based on the paper: https://arxiv.org/abs/2103.06508

    Args:
        embedding_dim: Encoder embedding dimension (default: 512)
        projection_dim: Projection head output dimension (default: 256)
        temperature: Temperature for InfoNCE loss (default: 0.07)
        lr: Learning rate (default: 1e-3)
        weight_decay: Weight decay for AdamW (default: 1e-4)
        max_epochs: Maximum epochs for cosine annealing (default: 100)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        projection_dim: int = 256,
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

        # Encoders
        self.encoder_1d = Encoder1D(embedding_dim=embedding_dim)
        self.encoder_2d = Encoder2D(embedding_dim=embedding_dim)

        # Projection heads
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

        # Loss function
        self.criterion = InfoNCELoss(temperature=temperature)

    def forward(self, waveform: torch.Tensor, spectrogram: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            waveform: Raw waveform tensor (batch, time)
            spectrogram: Spectrogram tensor (batch, freq, time)

        Returns:
            Tuple of (z1, z2) - projected embeddings from both encoders
        """
        # Encode
        h1 = self.encoder_1d(waveform)
        h2 = self.encoder_2d(spectrogram)

        # Project
        z1 = self.projection_head_1d(h1)
        z2 = self.projection_head_2d(h2)

        return z1, z2

    def get_embeddings(self, waveform: torch.Tensor, spectrogram: torch.Tensor) -> tuple:
        """Get embeddings without projection (for linear probe).

        Args:
            waveform: Raw waveform tensor (batch, time)
            spectrogram: Spectrogram tensor (batch, freq, time)

        Returns:
            Tuple of (h1, h2) - embeddings from encoders (before projection)
        """
        h1 = self.encoder_1d(waveform)
        h2 = self.encoder_2d(spectrogram)
        return h1, h2

    def training_step(self, batch, batch_idx):
        """Training step."""
        waveform = batch['waveform']
        spectrogram = batch['spectrogram']

        # Forward pass
        z1, z2 = self(waveform, spectrogram)

        # Compute contrastive loss
        loss = self.criterion(z1, z2)

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        waveform = batch['waveform']
        spectrogram = batch['spectrogram']

        # Forward pass
        z1, z2 = self(waveform, spectrogram)

        # Compute contrastive loss
        loss = self.criterion(z1, z2)

        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
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
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


class FinetuneModel(pl.LightningModule):
    """Fine-tune contrastive model for digit classification.

    Args:
        contrastive_model: Pre-trained contrastive model
        encoder_type: Which encoder to use ('1d', '2d', or 'concat')
        num_classes: Number of output classes (default: 10)
        freeze_encoder: Whether to freeze encoder weights (default: False)
        lr: Learning rate (default: 1e-4)
        weight_decay: Weight decay (default: 1e-4)
        max_epochs: Maximum epochs (default: 50)
    """

    def __init__(
        self,
        contrastive_model: MultiFormatContrastiveModel,
        encoder_type: str = '1d',
        num_classes: int = 10,
        freeze_encoder: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['contrastive_model'])

        self.encoder_type = encoder_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # Copy encoders from contrastive model
        self.encoder_1d = contrastive_model.encoder_1d
        self.encoder_2d = contrastive_model.encoder_2d

        # Freeze encoders if requested
        if freeze_encoder:
            for param in self.encoder_1d.parameters():
                param.requires_grad = False
            for param in self.encoder_2d.parameters():
                param.requires_grad = False

        # Determine classifier input dimension
        if encoder_type == 'concat':
            embedding_dim = 512 * 2  # Concatenated
        else:
            embedding_dim = 512

        # Classifier head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        from torchmetrics import Accuracy
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, waveform: torch.Tensor, spectrogram: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            waveform: Raw waveform tensor
            spectrogram: Spectrogram tensor

        Returns:
            Logits (batch, num_classes)
        """
        if self.encoder_type == '1d':
            features = self.encoder_1d(waveform)
        elif self.encoder_type == '2d':
            features = self.encoder_2d(spectrogram)
        elif self.encoder_type == 'concat':
            h1 = self.encoder_1d(waveform)
            h2 = self.encoder_2d(spectrogram)
            features = torch.cat([h1, h2], dim=1)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        """Training step."""
        waveform = batch['waveform']
        spectrogram = batch['spectrogram']
        y = batch['digit']

        logits = self(waveform, spectrogram)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        waveform = batch['waveform']
        spectrogram = batch['spectrogram']
        y = batch['digit']

        logits = self(waveform, spectrogram)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        waveform = batch['waveform']
        spectrogram = batch['spectrogram']
        y = batch['digit']

        logits = self(waveform, spectrogram)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)

        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/acc', acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
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
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
