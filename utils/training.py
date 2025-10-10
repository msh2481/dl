"""
Training loop utilities.
"""

import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import accuracy_score
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """Trainer class for supervised learning."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        save_every_n_steps: int = 100,
        early_stopping_patience: int = 10,
        use_wandb: bool = True,
        ema_span: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_every_n_steps = save_every_n_steps
        self.early_stopping_patience = early_stopping_patience
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.ema_span = ema_span

        self.best_val_acc = 0.0
        self.best_val_acc_ema = 0.0
        self.best_val_acc_polyak = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self.scheduler = None

        # Initialize EMA and Polyak models
        self.ema_model = copy.deepcopy(model)
        self.polyak_model = copy.deepcopy(model)
        self.polyak_count = 0  # Number of updates for Polyak averaging

    def set_scheduler(self, scheduler):
        """Set learning rate scheduler."""
        self.scheduler = scheduler

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update EMA and Polyak models after optimizer step
            self._update_averaged_models()

            # Track metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

            # Save checkpoint every N steps
            self.global_step += 1
            if self.global_step % self.save_every_n_steps == 0:
                step_checkpoint_path = self.checkpoint_dir / f"step_{self.global_step}.pt"
                torch.save(self.model.state_dict(), step_checkpoint_path)

            # Log to wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": current_lr,
                        "global_step": self.global_step,
                    }
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_targets, all_preds)

        return avg_loss, accuracy

    def _update_averaged_models(self):
        """Update EMA and Polyak averaged models."""
        # EMA update: decay = 2 / (span + 1)
        ema_decay = 2.0 / (self.ema_span + 1.0)

        for ema_param, polyak_param, param in zip(
            self.ema_model.parameters(),
            self.polyak_model.parameters(),
            self.model.parameters(),
        ):
            # EMA: ema = ema * (1 - decay) + param * decay
            ema_param.data.mul_(1.0 - ema_decay).add_(param.data, alpha=ema_decay)

            # Polyak: simple arithmetic mean
            self.polyak_count += 1
            polyak_param.data.mul_((self.polyak_count - 1) / self.polyak_count).add_(
                param.data, alpha=1.0 / self.polyak_count
            )

    @torch.no_grad()
    def validate(self, model: Optional[nn.Module] = None, desc: str = "Validating") -> tuple[float, float]:
        """Validate on validation set."""
        if model is None:
            model = self.model

        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=desc)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_targets, all_preds)

        return avg_loss, accuracy

    def train(self, num_epochs: int):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Training on {len(self.train_loader.dataset)} samples")
        logger.info(f"Validating on {len(self.val_loader.dataset)} samples")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate on current model
            val_loss, val_acc = self.validate(self.model, desc="Val [Current]")

            # Validate on EMA model
            val_loss_ema, val_acc_ema = self.validate(self.ema_model, desc="Val [EMA]")

            # Validate on Polyak model
            val_loss_polyak, val_acc_polyak = self.validate(self.polyak_model, desc="Val [Polyak]")

            # Log epoch metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} / EMA: {val_acc_ema:.4f} / Polyak: {val_acc_polyak:.4f}"
            )

            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/epoch_loss": train_loss,
                        "train/epoch_acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "val/loss_ema": val_loss_ema,
                        "val/acc_ema": val_acc_ema,
                        "val/loss_polyak": val_loss_polyak,
                        "val/acc_polyak": val_acc_polyak,
                    }
                )

            # Save best checkpoint for current model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_checkpoint_path = self.checkpoint_dir / "best.pt"
                torch.save(self.model.state_dict(), best_checkpoint_path)
                logger.info(f"Saved best checkpoint (current) with val_acc: {val_acc:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save best checkpoint for EMA model
            if val_acc_ema > self.best_val_acc_ema:
                self.best_val_acc_ema = val_acc_ema
                best_ema_checkpoint_path = self.checkpoint_dir / "best_ema.pt"
                torch.save(self.ema_model.state_dict(), best_ema_checkpoint_path)
                logger.info(f"Saved best EMA checkpoint with val_acc: {val_acc_ema:.4f}")

            # Save best checkpoint for Polyak model
            if val_acc_polyak > self.best_val_acc_polyak:
                self.best_val_acc_polyak = val_acc_polyak
                best_polyak_checkpoint_path = self.checkpoint_dir / "best_polyak.pt"
                torch.save(self.polyak_model.state_dict(), best_polyak_checkpoint_path)
                logger.info(f"Saved best Polyak checkpoint with val_acc: {val_acc_polyak:.4f}")

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.warning(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(patience: {self.early_stopping_patience})"
                )
                break

            # Step scheduler (per-epoch)
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final checkpoints
        final_checkpoint_path = self.checkpoint_dir / "final.pt"
        torch.save(self.model.state_dict(), final_checkpoint_path)

        final_ema_checkpoint_path = self.checkpoint_dir / "final_ema.pt"
        torch.save(self.ema_model.state_dict(), final_ema_checkpoint_path)

        final_polyak_checkpoint_path = self.checkpoint_dir / "final_polyak.pt"
        torch.save(self.polyak_model.state_dict(), final_polyak_checkpoint_path)

        logger.info(f"Training complete.")
        logger.info(f"Best val_acc (current): {self.best_val_acc:.4f}")
        logger.info(f"Best val_acc (EMA): {self.best_val_acc_ema:.4f}")
        logger.info(f"Best val_acc (Polyak): {self.best_val_acc_polyak:.4f}")
