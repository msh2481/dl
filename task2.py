from pathlib import Path
import json
import torch
import typer
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from audio_data import AudioMNISTDataModule
from audio_models import MultiFormatContrastiveModel
from audio_utils import LinearProbe

app = typer.Typer()


@app.command()
def main(
    ratio: float = 1.0,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_workers: int = 4,
):
    output_dir = Path("task2_contrastive")
    output_dir.mkdir(exist_ok=True)

    dm = AudioMNISTDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        ratio=ratio,
    )
    dm.setup()

    print(f"\n{'='*60}")
    print("Training Multi-Format Contrastive Model")
    print(f"{'='*60}\n")

    model = MultiFormatContrastiveModel(
        embedding_dim=128,
        projection_dim=128,
        temperature=0.07,
        lr=lr,
        max_epochs=epochs,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    early_stopping = EarlyStopping(monitor="val/loss", patience=10, mode="min")

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
    )

    trainer.fit(model, dm)

    print(f"\n{'='*60}")
    print("Evaluating with Linear Probes")
    print(f"{'='*60}\n")

    best_model = MultiFormatContrastiveModel.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )

    results = LinearProbe.evaluate_multimodal(
        best_model,
        dm.train_dataloader(),
        dm.val_dataloader(),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Clean results for JSON serialization (remove model objects)
    json_results = {}
    for feature_type, result in results.items():
        json_results[feature_type] = {
            "train_acc": result["train_acc"],
            "val_acc": result["val_acc"]
            # Exclude 'model' key as LogisticRegression is not JSON serializable
        }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(json_results, f, indent=2)

    with open(output_dir / "metrics.txt", "w") as f:
        f.write("Task 2: Multi-Format Contrastive Learning\n")
        f.write("=" * 40 + "\n\n")
        f.write("Linear Probe Results:\n")
        for probe_type, metrics in results.items():
            f.write(f"\n{probe_type.upper()} Probe:\n")
            f.write(f"  Validation Accuracy: {metrics['val_acc']:.4f}\n")
            f.write(f"  Test Accuracy:       {metrics['test_acc']:.4f}\n")

    print(f"\nResults saved to {output_dir}/")
    print("\nLinear Probe Results:")
    for probe_type, metrics in results.items():
        print(f"  {probe_type.upper()}: {metrics['test_acc']:.4f}")


if __name__ == "__main__":
    app()
