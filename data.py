from typing import Callable

import numpy as np
import torch
from beartype import beartype as typed
from datasets import Dataset, load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@typed
def get_datasets(fraction: float = 1.0) -> tuple[dict[str, Dataset], list[str]]:
    def sample(dataset):
        return dataset.shuffle().take(int(fraction * len(dataset)))

    full_dataset = load_dataset("wltjr1007/DomainNet")
    train, test = full_dataset["train"], full_dataset["test"]
    classnames = train.features["label"].names
    domain_ids = {
        "infograph": 1,
        "painting": 2,
        "quickdraw": 3,
        "real": 4,
        "clipart": 5,
    }
    iid_id = domain_ids["real"]

    results = {
        "ID": sample(
            train.filter(lambda x: np.array(x["domain"]) == iid_id, batched=True)
        )
    }
    for name, id in domain_ids.items():
        results[f"OOD_{name}"] = sample(
            test.filter(lambda x: np.array(x["domain"]) == id, batched=True)
        )
    return results, classnames


@typed
def get_dataloaders(
    datasets: dict[str, Dataset], preprocess: Callable, batch_size: int = 32
) -> dict[str, DataLoader]:
    def get_dataloader(dataset: Dataset) -> DataLoader:
        def hf_transform(example):
            if isinstance(example["image"], list):
                image = [preprocess(img) for img in example["image"]]
            else:
                image = preprocess(example["image"])
            return {"image": image, "label": example["label"]}

        processed = dataset.with_transform(hf_transform)
        return DataLoader(processed, batch_size=batch_size, shuffle=True)

    return {name: get_dataloader(dataset) for name, dataset in datasets.items()}


@typed
def evaluate(model: nn.Module, dataloaders: dict[str, DataLoader]) -> dict[str, float]:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    results = {}
    with torch.inference_mode():
        for name, dataloader in dataloaders.items():
            correct = 0
            total = 0
            for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                correct += (logits.argmax(dim=-1) == labels).float().sum()
                total += len(labels)
            results[name] = correct / total
    return results
