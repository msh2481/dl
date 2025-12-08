from pathlib import Path
import json
import typer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from audio_data import AudioMNISTDataModule
from audio_models import MultiFormatContrastiveModel
from audio_utils import extract_contrastive_embeddings

app = typer.Typer()


def evaluate_knn_speaker_identification(
    train_embeddings,
    train_speakers,
    test_embeddings,
    test_speakers,
    k=5,
):
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_embeddings, train_speakers)

    val_preds = knn.predict(test_embeddings)
    val_acc = accuracy_score(test_speakers, val_preds)

    return val_acc, knn


@app.command()
def main(
    ratio: float = 1.0,
    batch_size: int = 64,
    num_workers: int = 4,
    k: int = 5,
    task2_checkpoint: str = "task2_contrastive/checkpoints/best.ckpt",
    epochs: int = 0,
):
    output_dir = Path("task4_biometrics")
    output_dir.mkdir(exist_ok=True)

    dm = AudioMNISTDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        ratio=ratio,
    )
    dm.setup()

    print(f"\n{'='*60}")
    print("Loading Contrastive Model")
    print(f"{'='*60}\n")

    model = MultiFormatContrastiveModel.load_from_checkpoint(task2_checkpoint)

    print(f"\n{'='*60}")
    print("Extracting Embeddings")
    print(f"{'='*60}\n")

    train_emb_1d, train_emb_2d, train_speakers, _ = (
        extract_contrastive_embeddings(model, dm.train_dataloader())
    )
    val_emb_1d, val_emb_2d, val_speakers, _ = extract_contrastive_embeddings(
        model, dm.val_dataloader()
    )
    test_emb_1d, test_emb_2d, test_speakers, _ = extract_contrastive_embeddings(
        model, dm.test_dataloader()
    )

    train_emb_concat = np.concatenate([train_emb_1d, train_emb_2d], axis=1)
    val_emb_concat = np.concatenate([val_emb_1d, val_emb_2d], axis=1)
    test_emb_concat = np.concatenate([test_emb_1d, test_emb_2d], axis=1)

    print(f"\n{'='*60}")
    print(f"Training k-NN Speaker Identification (k={k})")
    print(f"{'='*60}\n")

    results = {}

    for emb_type, train_emb, val_emb, test_emb in [
        ("1d", train_emb_1d, val_emb_1d, test_emb_1d),
        ("2d", train_emb_2d, val_emb_2d, test_emb_2d),
        ("concat", train_emb_concat, val_emb_concat, test_emb_concat),
    ]:
        print(f"\nEvaluating {emb_type.upper()} embeddings...")

        val_acc, knn = evaluate_knn_speaker_identification(
            train_emb, train_speakers, val_emb, val_speakers, k=k
        )

        test_preds = knn.predict(test_emb)
        test_acc = accuracy_score(test_speakers, test_preds)

        results[emb_type] = {
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "k": k,
        }

        print(f"  Val Acc:  {val_acc:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "metrics.txt", "w") as f:
        f.write("Task 4: Voice Biometrics (Speaker Identification)\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"k-NN with k={k}\n\n")
        for emb_type, metrics in results.items():
            f.write(f"{emb_type.upper()} Embeddings:\n")
            f.write(f"  Validation Accuracy: {metrics['val_acc']:.4f}\n")
            f.write(f"  Test Accuracy:       {metrics['test_acc']:.4f}\n")
            f.write("\n")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    app()
