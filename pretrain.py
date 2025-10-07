from pathlib import Path

import torch
from loguru import logger
from torchvision import models


def main():
    # Create ResNet-18 with 10 classes
    model = models.resnet18(num_classes=10)

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Save the model
    checkpoint_path = checkpoint_dir / "pretrained_baseline.pt"
    torch.save(model.state_dict(), checkpoint_path)

    logger.info(f"Saved baseline ResNet-18 model to {checkpoint_path}")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")


if __name__ == "__main__":
    main()
