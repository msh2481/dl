import argparse
import copy
import os
from pathlib import Path
from itertools import chain
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.linear_model import LogisticRegression
from torchvision import models
from tqdm import tqdm

from contrastive_data import ContrastiveDataModule


class BYOL(pl.LightningModule):
    def __init__(self, hidden_dim: int = 128, lr: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs: int = 500, tau: float = 0.996):
        super().__init__()
        self.save_hyperparameters()

        self.online_backbone = models.resnet18(weights=None)
        num_ftrs = self.online_backbone.fc.in_features
        self.online_backbone.fc = nn.Identity()

        self.online_projector = nn.Sequential(
            nn.Linear(num_ftrs, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

        self.fc = nn.Linear(512, 10)
        self.fc.requires_grad_(False)

        self.target_backbone = copy.deepcopy(self.online_backbone)
        self.target_projector = copy.deepcopy(self.online_projector)

        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward(self, x):
        feats = self.online_backbone(x)
        z = self.online_projector(feats)
        return self.predictor(z)

    @torch.no_grad()
    def update_target_network(self):
        tau = self.hparams.tau
        for online_params, target_params in zip(
            chain(self.online_backbone.parameters(), self.online_projector.parameters()),
            chain(self.target_backbone.parameters(), self.target_projector.parameters())
        ):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data

    def byol_loss(self, batch):
        imgs, _ = batch
        view1, view2 = imgs[0], imgs[1]

        z1 = self.online_projector(self.online_backbone(view1))
        pred1 = self.predictor(z1)
        pred1 = F.normalize(pred1, dim=-1)

        z2 = self.online_projector(self.online_backbone(view2))
        pred2 = self.predictor(z2)
        pred2 = F.normalize(pred2, dim=-1)

        with torch.no_grad():
            target1 = self.target_projector(self.target_backbone(view1))
            target1 = F.normalize(target1, dim=-1)

            target2 = self.target_projector(self.target_backbone(view2))
            target2 = F.normalize(target2, dim=-1)

        loss1 = 2 - 2 * (pred1 * target2).sum(dim=-1).mean()
        loss2 = 2 - 2 * (pred2 * target1).sum(dim=-1).mean()

        z = torch.cat([z1, z2], dim=0)
        z_std = z.std(dim=0).mean()

        return (loss1 + loss2) / 2, z_std

    def training_step(self, batch, batch_idx):
        loss, z_std = self.byol_loss(batch)
        self.log('train_loss', loss, prog_bar=True)
        self.log('z_std', z_std, prog_bar=True)
        return loss

    def on_after_backward(self):
        self.update_target_network()

    @torch.no_grad()
    def on_train_epoch_end(self):
        self.online_backbone.eval()

        train_loader = self.trainer.datamodule.train_labeled_dataloader()
        test_loader = self.trainer.datamodule.test_dataloader()

        train_feats, train_labels = [], []
        for imgs, labels in tqdm(train_loader, desc="Train features"):
            feats = self.online_backbone(imgs.to(self.device))
            train_feats.append(feats.cpu())
            train_labels.append(labels)
        train_feats = torch.cat(train_feats).numpy()
        train_labels = torch.cat(train_labels).numpy()

        test_feats, test_labels = [], []
        for imgs, labels in tqdm(test_loader, desc="Test features"):
            feats = self.online_backbone(imgs.to(self.device))
            test_feats.append(feats.cpu())
            test_labels.append(labels)
        test_feats = torch.cat(test_feats).numpy()
        test_labels = torch.cat(test_labels).numpy()

        # Downsample 5x for faster logreg training
        n_samples = len(train_feats) // 5
        indices = torch.randperm(len(train_feats))[:n_samples].numpy()
        train_feats_sub = train_feats[indices]
        train_labels_sub = train_labels[indices]

        clf = LogisticRegression(max_iter=100, solver='lbfgs')
        clf.fit(train_feats_sub, train_labels_sub)

        train_acc = clf.score(train_feats, train_labels)
        test_acc = clf.score(test_feats, test_labels)

        self.log('train_logreg_acc', train_acc, prog_bar=True)
        self.log('test_logreg_acc', test_acc, prog_bar=True)

        # Save logreg weights to fc layer
        self.fc.weight.data = torch.from_numpy(clf.coef_).float()
        self.fc.bias.data = torch.from_numpy(clf.intercept_).float()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.996)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--epoch-ratio", type=float, default=1.0)
    args = parser.parse_args()

    args.epochs = int(args.epochs * args.epoch_ratio + 0.5)

    torch.set_float32_matmul_precision('medium')

    Path("checkpoints").mkdir(exist_ok=True)

    data_module = ContrastiveDataModule(args.data_dir, args.batch_size, args.num_workers)
    model = BYOL(args.hidden_dim, args.lr, args.weight_decay, args.epochs, args.tau)

    logger = WandbLogger(project="stl10-byol", log_model=False) if "WANDB_API_KEY" in os.environ else None

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="byol-best-{epoch:02d}-{test_logreg_acc:.4f}",
        monitor="test_logreg_acc",
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
        limit_train_batches=args.limit_train_batches,
    )

    trainer.fit(model, data_module)

    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best test accuracy: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
