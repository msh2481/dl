import argparse
import os
from pathlib import Path

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


class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim: int = 128, lr: float = 1e-3, temperature: float = 0.07,
                 weight_decay: float = 1e-4, max_epochs: int = 500):
        super().__init__()
        self.save_hyperparameters()
        assert temperature > 0.0, 'Temperature must be positive'

        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

        self.fc = nn.Linear(512, 10)
        self.fc.requires_grad_(False)

    def forward(self, x):
        feats = self.backbone(x)
        return self.projection_head(feats)

    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        feats = self(imgs)
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        self.log(f'{mode}_loss', nll, prog_bar=True)

        combined_sim = torch.cat([cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
        sim_argsort = combined_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        self.log(f'{mode}_acc_top1', (sim_argsort == 0).float().mean(), prog_bar=(mode == 'train'))
        self.log(f'{mode}_acc_top5', (sim_argsort < 5).float().mean())
        self.log(f'{mode}_acc_mean_pos', 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    @torch.no_grad()
    def on_train_epoch_end(self):
        self.backbone.eval()

        train_loader = self.trainer.datamodule.train_labeled_dataloader()
        test_loader = self.trainer.datamodule.test_dataloader()

        train_feats, train_labels = [], []
        for imgs, labels in tqdm(train_loader, desc="Train features"):
            feats = self.backbone(imgs.to(self.device))
            train_feats.append(feats.cpu())
            train_labels.append(labels)
        train_feats = torch.cat(train_feats).numpy()
        train_labels = torch.cat(train_labels).numpy()

        test_feats, test_labels = [], []
        for imgs, labels in tqdm(test_loader, desc="Test features"):
            feats = self.backbone(imgs.to(self.device))
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
    parser.add_argument("--temperature", type=float, default=0.07)
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
    model = SimCLR(args.hidden_dim, args.lr, args.temperature, args.weight_decay, args.epochs)

    logger = WandbLogger(project="stl10-simclr", log_model=False) if "WANDB_API_KEY" in os.environ else None

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="simclr-best-{epoch:02d}-{test_logreg_acc:.4f}",
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
