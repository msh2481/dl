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
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import STL10
from tqdm import tqdm


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim: int = 128, lr: float = 1e-3, temperature: float = 0.07,
                 weight_decay: float = 1e-4, max_epochs: int = 500):
        super().__init__()
        self.save_hyperparameters()
        assert temperature > 0.0, 'Temperature must be positive'

        self.convnet = models.resnet18(weights=None)
        self.convnet.fc = nn.Sequential(
            nn.Linear(self.convnet.fc.in_features, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.convnet(x)

    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        feats = self.convnet(imgs)
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
        self.convnet.eval()

        train_loader = self.trainer.datamodule.train_labeled_dataloader()
        test_loader = self.trainer.datamodule.test_dataloader()

        train_feats, train_labels = [], []
        for imgs, labels in tqdm(train_loader, desc="Train features"):
            feats = self.convnet(imgs.to(self.device))
            train_feats.append(feats.cpu())
            train_labels.append(labels)
        train_feats = torch.cat(train_feats).numpy()
        train_labels = torch.cat(train_labels).numpy()

        test_feats, test_labels = [], []
        for imgs, labels in tqdm(test_loader, desc="Test features"):
            feats = self.convnet(imgs.to(self.device))
            test_feats.append(feats.cpu())
            test_labels.append(labels)
        test_feats = torch.cat(test_feats).numpy()
        test_labels = torch.cat(test_labels).numpy()

        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        clf.fit(train_feats, train_labels)

        train_acc = clf.score(train_feats, train_labels)
        test_acc = clf.score(test_feats, test_labels)

        self.log('train_logreg_acc', train_acc, prog_bar=True)
        self.log('test_logreg_acc', test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class SimCLRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 256, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        normalize = transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713])

        contrast_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=96),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            normalize,
        ])

        self.train_transform = ContrastiveTransformations(contrast_transforms, n_views=2)
        self.eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def setup(self, stage=None):
        self.train_dataset = STL10(self.data_dir, split="unlabeled", download=True, transform=self.train_transform)
        self.train_labeled = STL10(self.data_dir, split="train", download=True, transform=self.eval_transform)
        self.test_dataset = STL10(self.data_dir, split="test", download=True, transform=self.eval_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def train_labeled_dataloader(self):
        return DataLoader(
            self.train_labeled,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    args = parser.parse_args()

    Path("checkpoints").mkdir(exist_ok=True)

    data_module = SimCLRDataModule(args.data_dir, args.batch_size, args.num_workers)
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
