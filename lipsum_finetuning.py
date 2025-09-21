import numpy as np
import torch
import typer
from clip.clip import _tokenizer
from loguru import logger
from torch import nn
from tqdm import tqdm

from data import get_dataloaders, get_datasets
from random_texts import CLIPZeroShotClassifier


def save_model(model: torch.nn.Module, filepath: str) -> None:
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")


def sample_random_tokens(n: int, L: int = 8) -> list[str]:
    V = len(_tokenizer.encoder)
    return [
        "".join(_tokenizer.decode(np.random.randint(0, V, size=L))) for _ in range(n)
    ]


def main(
    fraction: float = typer.Option(1e-3, help="Fraction of dataset to use"),
    batch_size: int = typer.Option(128, help="Batch size for training"),
    lr: float = typer.Option(1e-5, help="Learning rate"),
    weight_decay: float = typer.Option(0.1, help="Weight decay for optimizer"),
    warmup_fraction: float = typer.Option(
        0.1, help="Fraction of total steps for warmup"
    ),
    lambda_lipsum: float = typer.Option(0.1, help="Weight for lipsum loss"),
    max_grad_norm: float = typer.Option(1.0, help="Maximum gradient norm for clipping"),
    use_float32: bool = typer.Option(
        True, help="Use float32 instead of float16 for stability"
    ),
) -> None:
    logger.info(
        f"Starting lipsum fine-tuning with fraction: {fraction}, batch size: {batch_size}, lr: {lr}, lambda: {lambda_lipsum}"
    )

    datasets, classnames = get_datasets(fraction=fraction)
    for name, dataset in datasets.items():
        logger.info(
            f"{name}: {len(dataset)} samples -> {(len(dataset) + batch_size - 1) // batch_size} batches"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    baseline_model = CLIPZeroShotClassifier(classnames, use_float32=use_float32)
    lipsum_model = CLIPZeroShotClassifier(classnames, use_float32=use_float32)
    logger.info(f"Model dtype: {lipsum_model.dtype}, device: {lipsum_model.device}")
    dataloaders = get_dataloaders(
        datasets, lipsum_model.preprocess, batch_size=batch_size
    )
    logger.info("Dataloaders created")

    optimizer = torch.optim.AdamW(
        lipsum_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = len(dataloaders["ID"])
    warmup_steps = int(warmup_fraction * total_steps)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    logger.info(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

    pbar = tqdm(dataloaders["ID"], desc="Fine-tuning")
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Check for NaN/inf in inputs
        if torch.isnan(images).any() or torch.isinf(images).any():
            logger.error(f"NaN/Inf detected in images at step {step}")
            continue
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            logger.error(f"NaN/Inf detected in labels at step {step}")
            continue

        texts = sample_random_tokens(len(images))
        logits = lipsum_model(images)

        # Check for NaN/inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error(f"NaN/Inf detected in logits at step {step}")
            continue

        cur_energy = lipsum_model.get_energy(images, texts)
        with torch.no_grad():
            old_energy = baseline_model.get_energy(images, texts)

        # Check energies for NaN/inf
        if torch.isnan(cur_energy).any() or torch.isinf(cur_energy).any():
            logger.error(f"NaN/Inf detected in current energy at step {step}")
            continue
        if torch.isnan(old_energy).any() or torch.isinf(old_energy).any():
            logger.error(f"NaN/Inf detected in old energy at step {step}")
            continue

        ce_loss = nn.functional.cross_entropy(logits, labels)
        gap_loss = nn.functional.mse_loss(cur_energy, old_energy)
        loss = ce_loss + lambda_lipsum * gap_loss

        # Check losses for NaN/inf
        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            logger.error(f"NaN/Inf CE loss detected at step {step}")
            continue
        if torch.isnan(gap_loss) or torch.isinf(gap_loss):
            logger.error(f"NaN/Inf gap loss detected at step {step}")
            continue
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN/Inf total loss detected at step {step}")
            continue

        if step % 10 == 0:
            logger.info(
                f"Step {step}, CE Loss: {ce_loss.item():.4f}, Gap Loss: {gap_loss.item():.4f}, Total: {loss.item():.4f}"
            )
            save_model(lipsum_model, "lipsum_model.pth")

        optimizer.zero_grad()
        loss.backward()

        # Check gradients before clipping
        total_norm = torch.nn.utils.clip_grad_norm_(
            lipsum_model.parameters(), max_grad_norm, norm_type="inf"
        )
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            logger.error(f"NaN/Inf gradient norm detected at step {step}: {total_norm}")
            continue

        pbar.set_postfix(
            ce_loss=f"{ce_loss.item():.4f}",
            gap_loss=f"{gap_loss.item():.4f}",
            grad_norm=f"{total_norm:.3f}",
        )

        optimizer.step()
        scheduler.step()

    save_model(lipsum_model, "lipsum_model.pth")
    logger.success("Lipsum fine-tuning completed")


if __name__ == "__main__":
    typer.run(main)
