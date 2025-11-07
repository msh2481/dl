import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from checkpoint_utils import load_model_for_classification


# STL-10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
# CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# Mapping: CIFAR-10 -> STL-10 (skip frog which is index 6)
CIFAR_TO_STL_MAPPING = {
    0: 0,  # airplane
    1: 2,  # automobile -> car
    2: 1,  # bird
    3: 3,  # cat
    4: 4,  # deer
    5: 5,  # dog
    6: 7,  # frog -> monkey (no other appropriate mapping)
    7: 6,  # horse
    8: 8,  # ship
    9: 9,  # truck
}


def evaluate_on_cifar10(model, device, data_dir="./data", batch_size=256, num_workers=4):
    """Evaluate model on CIFAR-10 test set"""
    model.eval()
    model.to(device)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = CIFAR10(data_dir, train=False, download=True, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Evaluating"):
            imgs = imgs.to(device)

            # Map CIFAR-10 labels to STL-10 labels
            labels = torch.tensor([CIFAR_TO_STL_MAPPING[l.item()] for l in labels])

            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint}")
    model = load_model_for_classification(args.checkpoint, num_classes=10)
    model.to(device)

    print("Evaluating on CIFAR-10 (excluding frogs)...")
    accuracy = evaluate_on_cifar10(model, device, args.data_dir, args.batch_size, args.num_workers)

    print(f"\nOOD Accuracy on CIFAR-10: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()
