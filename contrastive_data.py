import lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import STL10


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class ContrastiveDataModule(pl.LightningDataModule):
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
