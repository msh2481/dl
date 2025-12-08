from pathlib import Path
import json
import typer
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from audio_data import AudioMNISTDataModule
from audio_models import MultiFormatContrastiveModel, FinetuneModel

app = typer.Typer()


@app.command()
def main(
    ratio: float = 1.0,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 4,
    task2_checkpoint: str = "task2_contrastive/checkpoints/best.ckpt",
):
    output_dir = Path("task5_finetune")
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

    contrastive_model = MultiFormatContrastiveModel.load_from_checkpoint(
        task2_checkpoint
    )

    results = {}

    for encoder_type in ["1d", "2d", "concat"]:
        for freeze in [False, True]:
            config_name = (
                f"{encoder_type}_{'frozen' if freeze else 'finetuned'}"
            )

            print(f"\n{'='*60}")
            print(
                f"Training {encoder_type.upper()} "
                f"({'Frozen' if freeze else 'Fine-tuned'})"
            )
            print(f"{'='*60}\n")

            model = FinetuneModel(
                contrastive_model=contrastive_model,
                encoder_type=encoder_type,
                freeze_encoder=freeze,
                lr=lr,
                max_epochs=epochs,
            )

            checkpoint_callback = ModelCheckpoint(
                dirpath=output_dir / f"{config_name}_checkpoints",
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

            results[config_name] = {
                "val_acc": float(
                    checkpoint_callback.best_model_score.cpu().numpy()
                ),
                "test_acc": test_results[0]["test/acc"],
                "test_loss": test_results[0]["test/loss"],
                "encoder_type": encoder_type,
                "frozen": freeze,
            }

            print(f"\n{config_name.upper()} Results:")
            print(f"  Val Acc:  {results[config_name]['val_acc']:.4f}")
            print(f"  Test Acc: {results[config_name]['test_acc']:.4f}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "metrics.txt", "w") as f:
        f.write("Task 5: Fine-tuning Pre-trained Models\n")
        f.write("=" * 40 + "\n\n")
        for config_name, metrics in results.items():
            f.write(f"{config_name.upper()}:\n")
            f.write(f"  Validation Accuracy: {metrics['val_acc']:.4f}\n")
            f.write(f"  Test Accuracy:       {metrics['test_acc']:.4f}\n")
            f.write(f"  Test Loss:           {metrics['test_loss']:.4f}\n")
            f.write("\n")

    print(f"\nResults saved to {output_dir}/")

    print("\n" + "=" * 60)
    print("Summary of All Configurations")
    print("=" * 60)
    for config_name, metrics in results.items():
        print(f"{config_name:25s}: {metrics['test_acc']:.4f}")


if __name__ == "__main__":
    app()
