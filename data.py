from typing import Callable

import numpy as np
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


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
