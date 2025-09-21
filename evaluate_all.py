import json
import os

import typer
from loguru import logger

from data import evaluate, get_dataloaders, get_datasets
from random_texts import CLIPZeroShotClassifier


def main(
    batch_size: int = typer.Option(128, help="Batch size for evaluation"),
    fraction: float = typer.Option(1e-3, help="Fraction of dataset to use"),
) -> None:
    logger.info(
        f"Starting evaluation with fraction: {fraction}, batch size: {batch_size}"
    )

    datasets, classnames = get_datasets(fraction=fraction)
    for name, dataset in datasets.items():
        logger.info(
            f"{name}: {len(dataset)} samples -> {(len(dataset) + batch_size - 1) // batch_size} batches"
        )

    models = {
        "baseline": CLIPZeroShotClassifier(classnames),
    }

    dataloaders = get_dataloaders(
        datasets,
        models["baseline"].preprocess,
        batch_size=batch_size,
    )

    os.makedirs("results", exist_ok=True)
    for model_name, model in models.items():
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
