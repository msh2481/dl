import json
import os
from pathlib import Path

import torch
import typer
from loguru import logger

from data import evaluate, get_dataloaders, get_datasets
from random_texts import CLIPZeroShotClassifier


def interpolate_models(
    model1: torch.nn.Module, model2: torch.nn.Module, alpha: float
) -> torch.nn.Module:
    """Linearly interpolate between two models: (1-alpha) * model1 + alpha * model2"""
    interpolated_state = {}
    state1 = model1.state_dict()
    state2 = model2.state_dict()

    for key in state1.keys():
        interpolated_state[key] = (1 - alpha) * state1[key] + alpha * state2[key]

    # Create new model with same architecture as model1
    interpolated_model = CLIPZeroShotClassifier(model1.classnames, use_float32=True)
    interpolated_model.load_state_dict(interpolated_state)
    return interpolated_model


def main(
    batch_size: int = typer.Option(128, help="Batch size for evaluation"),
    fraction: float = typer.Option(1e-3, help="Fraction of dataset to use"),
    ft_model_path: str = typer.Option("ft_model.pth", help="Path to fine-tuned model"),
    lipsum_model_path: str = typer.Option(
        "lipsum_model.pth", help="Path to lipsum model"
    ),
    n_points: int = typer.Option(10, help="Number of interpolation points"),
) -> None:
    logger.info(
        f"Starting evaluation with fraction: {fraction}, batch size: {batch_size}, n_points: {n_points}"
    )

    datasets, classnames = get_datasets(fraction=fraction)
    for name, dataset in datasets.items():
        logger.info(
            f"{name}: {len(dataset)} samples -> {(len(dataset) + batch_size - 1) // batch_size} batches"
        )

    # Create baseline model
    baseline_model = CLIPZeroShotClassifier(classnames, use_float32=True)
    logger.info("Created baseline model")

    models_to_evaluate = {}

    if Path(ft_model_path).exists():
        logger.info(f"Found {ft_model_path}, creating interpolations")
        ft_model = CLIPZeroShotClassifier(classnames, use_float32=True)
        ft_model.load_state_dict(torch.load(ft_model_path, map_location="cpu"))

        # Create interpolations
        for i in range(n_points + 1):
            alpha = i / n_points
            interpolated = interpolate_models(baseline_model, ft_model, alpha)
            model_name = f"ft_model_{i}_{n_points}"
            models_to_evaluate[model_name] = interpolated
            logger.info(f"Created {model_name} with alpha={alpha:.2f}")
    else:
        logger.warning(f"{ft_model_path} not found, skipping ft_model interpolations")

    # Check for lipsum_model.pth
    if Path(lipsum_model_path).exists():
        logger.info(f"Found {lipsum_model_path}, creating interpolations")
        lipsum_model = CLIPZeroShotClassifier(classnames, use_float32=True)
        lipsum_model.load_state_dict(torch.load(lipsum_model_path, map_location="cpu"))

        # Create interpolations
        for i in range(n_points + 1):
            alpha = i / n_points
            interpolated = interpolate_models(baseline_model, lipsum_model, alpha)
            model_name = f"lipsum_model_{i}_{n_points}"
            models_to_evaluate[model_name] = interpolated
            logger.info(f"Created {model_name} with alpha={alpha:.2f}")
    else:
        logger.warning(
            f"{lipsum_model_path} not found, skipping lipsum_model interpolations"
        )

    # Create dataloaders using baseline model's preprocessing
    dataloaders = get_dataloaders(
        datasets,
        baseline_model.preprocess,
        batch_size=batch_size,
    )

    os.makedirs("results", exist_ok=True)
    for model_name, model in models_to_evaluate.items():
        json_filename = f"results/{model_name}.json"
        if os.path.exists(json_filename):
            logger.info(f"Skipping {model_name} - {json_filename} already exists")
            continue
        logger.info(f"Evaluating {model_name}...")
        results = evaluate(model, dataloaders)
        logger.info(f"Results for {model_name}: {results}")
        results_serializable = {k: float(v) for k, v in results.items()}
        with open(json_filename, "w") as f:
            json.dump(results_serializable, f, indent=2)
        logger.success(f"Saved results to {json_filename}")


if __name__ == "__main__":
    typer.run(main)
