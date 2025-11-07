import argparse

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import STL10
from tqdm import tqdm

from checkpoint_utils import load_backbone_from_checkpoint


def extract_features(backbone, dataloader, device):
    """Extract features from backbone for all images in dataloader."""
    backbone.eval()
    backbone.to(device)

    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting features"):
            feats = backbone(imgs.to(device))
            features.append(feats.cpu())
            labels.append(lbls)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    return features, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "both"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading backbone from {args.checkpoint}")
    backbone = load_backbone_from_checkpoint(args.checkpoint)
    backbone.to(device)

    normalize = transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    splits = ["train", "test"] if args.split == "both" else [args.split]
    all_features, all_labels, all_split_labels = [], [], []

    for split in splits:
        dataset = STL10(args.data_dir, split=split, download=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        features, labels = extract_features(backbone, dataloader, device)
        all_features.append(features)
        all_labels.append(labels)
        all_split_labels.extend([split] * len(labels))

    all_features = torch.from_numpy(all_features[0]) if len(all_features) == 1 else torch.from_numpy(
        torch.cat([torch.from_numpy(f) for f in all_features]).numpy()
    )
    all_labels = all_labels[0] if len(all_labels) == 1 else torch.cat([torch.from_numpy(l) for l in all_labels]).numpy()

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings = tsne.fit_transform(all_features.numpy())

    print("Plotting...")
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(range(10))

    for class_idx in range(10):
        mask = all_labels == class_idx
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[colors[class_idx]],
            label=f"Class {class_idx}",
            alpha=0.6,
            s=10,
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"t-SNE Visualization ({args.split})")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
