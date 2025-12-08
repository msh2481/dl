from pathlib import Path
import torch
import typer
import lightning as pl

from audio_data import AudioMNISTDataModule
from audio_models import SupervisedModel, MultiFormatContrastiveModel
from audio_utils import generate_all_tsne_plots

app = typer.Typer()


@app.command()
def main(
    ratio: float = 1.0,
    batch_size: int = 64,
    num_workers: int = 4,
    task1_checkpoint: str = "task1_supervised",
    task2_checkpoint: str = "task2_contrastive/checkpoints/best.ckpt",
):
    output_dir = Path("task3_visualization")
    output_dir.mkdir(exist_ok=True)

    dm = AudioMNISTDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        ratio=ratio,
    )
    dm.setup()

    print(f"\n{'='*60}")
    print("Loading Models")
    print(f"{'='*60}\n")

    task1_dir = Path(task1_checkpoint)
    supervised_1d = SupervisedModel.load_from_checkpoint(
        str(task1_dir / "1d_checkpoints" / "best.ckpt")
    )
    supervised_2d = SupervisedModel.load_from_checkpoint(
        str(task1_dir / "2d_checkpoints" / "best.ckpt")
    )

    contrastive = MultiFormatContrastiveModel.load_from_checkpoint(
        task2_checkpoint
    )

    print(f"\n{'='*60}")
    print("Generating t-SNE Visualizations")
    print(f"{'='*60}\n")

    # Generate t-SNE plots for supervised models
    supervised_models = {
        "supervised_1d": (supervised_1d.encoder, "1d"),
        "supervised_2d": (supervised_2d.encoder, "2d"),
    }
    
    generate_all_tsne_plots(supervised_models, dm.val_dataloader(), output_dir)
    
    # Generate t-SNE plots for contrastive model (both 1d and 2d embeddings)
    from audio_utils.visualization import extract_contrastive_embeddings, plot_tsne
    import os
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for encoder_type in ["1d", "2d"]:
        print(f"\n{'='*60}")
        print(f"Generating t-SNE plots for contrastive_{encoder_type}")
        print(f"{'='*60}")
        
        embeddings, digits, speakers = extract_contrastive_embeddings(
            contrastive, dm.val_dataloader(), device, encoder_type
        )
        
        plot_tsne(
            embeddings=embeddings,
            labels=digits,
            title=f"contrastive_{encoder_type} - Colored by Digit",
            save_path=os.path.join(output_dir, f"contrastive_{encoder_type}_by_digit.png"),
            label_type="digit",
            perplexity=30,
        )
        
        plot_tsne(
            embeddings=embeddings,
            labels=speakers,
            title=f"contrastive_{encoder_type} - Colored by Speaker",
            save_path=os.path.join(output_dir, f"contrastive_{encoder_type}_by_speaker.png"),
            label_type="speaker",
            perplexity=30,
        )

    print(f"\nVisualizations saved to {output_dir}/")

    with open(output_dir / "metrics.txt", "w") as f:
        f.write("Task 3: t-SNE Visualization\n")
        f.write("=" * 40 + "\n\n")
        f.write("Generated 8 t-SNE plots:\n")
        f.write("  - supervised_1d_by_digit.png\n")
        f.write("  - supervised_1d_by_speaker.png\n")
        f.write("  - supervised_2d_by_digit.png\n")
        f.write("  - supervised_2d_by_speaker.png\n")
        f.write("  - contrastive_1d_by_digit.png\n")
        f.write("  - contrastive_1d_by_speaker.png\n")
        f.write("  - contrastive_2d_by_digit.png\n")
        f.write("  - contrastive_2d_by_speaker.png\n")


if __name__ == "__main__":
    app()
