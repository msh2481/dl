import json
import os
from glob import glob

from matplotlib import pyplot as plt

ft_results = []
lipsum_results = []

for name in glob("results/*.json"):
    with open(name, "r") as f:
        data = json.load(f)
        num = int(name.split("_")[-2])
        if "ft" in name:
            ft_results.append((num, data))
        else:
            lipsum_results.append((num, data))

ft_results.sort(key=lambda x: x[0])
lipsum_results.sort(key=lambda x: x[0])

ood_splits = ["infograph", "painting", "quickdraw", "real", "clipart"]
os.makedirs("fronts", exist_ok=True)
for ood_split in ood_splits:
    ft_results_iid = [x[1]["ID"] for x in ft_results]
    ft_results_ood = [x[1][f"OOD_{ood_split}"] for x in ft_results]
    lipsum_results_iid = [x[1]["ID"] for x in lipsum_results]
    lipsum_results_ood = [x[1][f"OOD_{ood_split}"] for x in lipsum_results]

    plt.figure(figsize=(8, 8))
    plt.title(f"ID vs OOD {ood_split.upper()}")
    plt.plot(ft_results_iid, ft_results_ood, label=f"FT", marker="D", lw=1)
    plt.plot(
        lipsum_results_iid,
        lipsum_results_ood,
        label=f"Lipsum-FT",
        marker="o",
        lw=1,
    )
    plt.legend()
    plt.xlabel("ID")
    plt.ylabel(f"OOD {ood_split.upper()}")
    plt.savefig(f"fronts/{ood_split}.png")
    plt.close()
