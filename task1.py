from pathlib import Path
import json
import typer
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from audio_data import AudioMNISTDataModule
from audio_models import SupervisedModel

app = typer.Typer()


@app.command()
def main(
    ratio: float = 1.0,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_workers: int = 4,
):
    output_dir = Path("task1_supervised")
    output_dir.mkdir(exist_ok=True)

    dm = AudioMNISTDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        ratio=ratio,
    )
    dm.setup()

    results = {}

    for encoder_type in ["1d", "2d"]:
        print(f"\n{'='*60}")
        print(f"Training {encoder_type.upper()} encoder")
        print(f"{'='*60}\n")

        model = SupervisedModel(
            encoder_type=encoder_type,
            embedding_dim=128,
            lr=lr,
            max_epochs=epochs,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir / f"{encoder_type}_checkpoints",
            filename="best",
            monitor="val/acc",
            mode="max",
            save_top_k=1,
        )

        early_stopping = EarlyStopping(
            monitor="val/acc", patience=10, mode="max"
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[checkpoint_callback, early_stopping],
            default_root_dir=str(output_dir),
            enable_progress_bar=True,
        )

        trainer.fit(model, dm)

        test_results = trainer.test(model, dm, ckpt_path="best")

        results[encoder_type] = {
            "val_acc": float(
                checkpoint_callback.best_model_score.cpu().numpy()
            ),
            "test_acc": test_results[0]["test/acc"],
            "test_loss": test_results[0]["test/loss"],
        }

        print(f"\n{encoder_type.upper()} Results:")
        print(f"  Val Acc:  {results[encoder_type]['val_acc']:.4f}")
        print(f"  Test Acc: {results[encoder_type]['test_acc']:.4f}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "metrics.txt", "w") as f:
        f.write("Task 1: Supervised Learning\n")
        f.write("=" * 40 + "\n\n")
        for encoder_type, metrics in results.items():
            f.write(f"{encoder_type.upper()} Encoder:\n")
            f.write(f"  Validation Accuracy: {metrics['val_acc']:.4f}\n")
            f.write(f"  Test Accuracy:       {metrics['test_acc']:.4f}\n")
            f.write(f"  Test Loss:           {metrics['test_loss']:.4f}\n")
            f.write("\n")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    app()
