import torch
import typer
from loguru import logger
from torch import nn
from tqdm import tqdm

from data import get_dataloaders, get_datasets
from random_texts import CLIPZeroShotClassifier


def save_model(model: torch.nn.Module, filepath: str) -> None:
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")


def main(
    fraction: float = typer.Option(1e-3, help="Fraction of dataset to use"),
    batch_size: int = typer.Option(128, help="Batch size for training"),
    lr: float = typer.Option(3e-5, help="Learning rate"),
    weight_decay: float = typer.Option(0.1, help="Weight decay for optimizer"),
    warmup_steps: int = typer.Option(500, help="Number of warmup steps"),
) -> None:
    logger.info(
        f"Starting fine-tuning with fraction: {fraction}, batch size: {batch_size}, lr: {lr}"
    )

    datasets, classnames = get_datasets(fraction=fraction)
    for name, dataset in datasets.items():
        logger.info(
            f"{name}: {len(dataset)} samples -> {(len(dataset) + batch_size - 1) // batch_size} batches"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ft_model = CLIPZeroShotClassifier(classnames)
    dataloaders = get_dataloaders(datasets, ft_model.preprocess, batch_size=batch_size)
    logger.info("Dataloaders created")

    optimizer = torch.optim.AdamW(
        ft_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = len(dataloaders["ID"])
    actual_warmup_steps = min(warmup_steps, total_steps // 2)
    logger.info(f"Total steps: {total_steps}, warmup steps: {actual_warmup_steps}")

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=actual_warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - actual_warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[actual_warmup_steps],
    )

    pbar = tqdm(dataloaders["ID"], desc="Fine-tuning")
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = ft_model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % 100 == 0:
            logger.info(f"Step {step}, Loss: {loss.item():.4f}")
            save_model(ft_model, "ft_model.pth")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    save_model(ft_model, "ft_model.pth")
    logger.success("Fine-tuning completed")


if __name__ == "__main__":
    typer.run(main)
