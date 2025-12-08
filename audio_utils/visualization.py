import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm


def plot_tsne(
    embeddings,
    labels,
    title,
    save_path,
    label_type="digit",
    perplexity=30,
    n_iter=1000,
    figsize=(10, 8),
):
    print(f"Computing t-SNE for {title}...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        verbose=0,
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    if label_type == "digit":
        n_classes = 10
        cmap = plt.cm.tab10
        label_name = "Digit"
    else:
        n_classes = len(np.unique(labels))
        cmap = plt.cm.tab20 if n_classes <= 20 else plt.cm.viridis
        label_name = "Speaker"

    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[cmap(label / n_classes)],
            label=str(label),
            alpha=0.6,
            s=30,
        )

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title)

    if label_type == "digit":
        ax.legend(title=label_name, loc="best", ncol=2)
    else:
        if n_classes <= 20:
            ax.legend(
                title=label_name,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                ncol=2,
            )
        else:
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_classes)
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(label_name)

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved t-SNE plot to {save_path}")


@torch.no_grad()
def extract_embeddings(encoder, dataloader, device="cuda", encoder_type="1d"):
    encoder.eval()
    encoder = encoder.to(device)

    all_embeddings = []
    all_digits = []
    all_speakers = []

    for batch in tqdm(dataloader, desc=f"Extracting embeddings"):
        if encoder_type == "1d":
            inputs = batch["waveform"].to(device)
        else:
            inputs = batch["spectrogram"].to(device)

        embeddings = encoder(inputs)

        all_embeddings.append(embeddings.cpu())
        all_digits.append(batch["digit"].cpu())
        all_speakers.append(batch["speaker_id"].cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    digits = torch.cat(all_digits, dim=0).numpy()
    speakers = torch.cat(all_speakers, dim=0).numpy()

    return embeddings, digits, speakers


@torch.no_grad()
def extract_contrastive_embeddings_both(
    contrastive_model,
    dataloader,
    device="cuda",
):
    """Extract both 1D and 2D embeddings from contrastive model"""
    contrastive_model.eval()
    contrastive_model = contrastive_model.to(device)

    all_embeddings_1d = []
    all_embeddings_2d = []
    all_digits = []
    all_speakers = []

    for batch in tqdm(dataloader, desc="Extracting both 1d and 2d embeddings"):
        waveform = batch["waveform"].to(device)
        spectrogram = batch["spectrogram"].to(device)

        h1, h2 = contrastive_model.get_embeddings(waveform, spectrogram)

        all_embeddings_1d.append(h1.cpu())
        all_embeddings_2d.append(h2.cpu())
        all_digits.append(batch["digit"].cpu())
        all_speakers.append(batch["speaker_id"].cpu())

    embeddings_1d = torch.cat(all_embeddings_1d, dim=0).numpy()
    embeddings_2d = torch.cat(all_embeddings_2d, dim=0).numpy()
    digits = torch.cat(all_digits, dim=0).numpy()
    speakers = torch.cat(all_speakers, dim=0).numpy()

    return embeddings_1d, embeddings_2d, speakers, digits


def generate_all_tsne_plots(
    models_dict,
    dataloader,
    save_dir="tsne_plots",
    device="cuda",
    perplexity=30,
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for model_name, (encoder, encoder_type) in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Generating t-SNE plots for {model_name}")
        print(f"{'='*60}")

        embeddings, digits, speakers = extract_embeddings(
            encoder, dataloader, device, encoder_type
        )

        plot_tsne(
            embeddings=embeddings,
            labels=digits,
            title=f"{model_name} - Colored by Digit",
            save_path=os.path.join(save_dir, f"{model_name}_by_digit.png"),
            label_type="digit",
            perplexity=perplexity,
        )

        plot_tsne(
            embeddings=embeddings,
            labels=speakers,
            title=f"{model_name} - Colored by Speaker",
            save_path=os.path.join(save_dir, f"{model_name}_by_speaker.png"),
            label_type="speaker",
            perplexity=perplexity,
        )

    print(f"\n{'='*60}")
    print(f"All t-SNE plots saved to {save_dir}")
    print(f"{'='*60}")


@torch.no_grad()
def extract_contrastive_embeddings(
    contrastive_model,
    dataloader,
    device="cuda",
    encoder_type="1d",
):
    contrastive_model.eval()
    contrastive_model = contrastive_model.to(device)

    all_embeddings = []
    all_digits = []
    all_speakers = []

    for batch in tqdm(dataloader, desc=f"Extracting {encoder_type} embeddings"):
        waveform = batch["waveform"].to(device)
        spectrogram = batch["spectrogram"].to(device)

        h1, h2 = contrastive_model.get_embeddings(waveform, spectrogram)

        if encoder_type == "1d":
            embeddings = h1
        else:
            embeddings = h2

        all_embeddings.append(embeddings.cpu())
        all_digits.append(batch["digit"].cpu())
        all_speakers.append(batch["speaker_id"].cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    digits = torch.cat(all_digits, dim=0).numpy()
    speakers = torch.cat(all_speakers, dim=0).numpy()

    return embeddings, digits, speakers


@torch.no_grad()
def extract_contrastive_embeddings_both(
    contrastive_model,
    dataloader,
    device="cuda",
):
    """Extract both 1D and 2D embeddings from contrastive model"""
    contrastive_model.eval()
    contrastive_model = contrastive_model.to(device)

    all_embeddings_1d = []
    all_embeddings_2d = []
    all_digits = []
    all_speakers = []

    for batch in tqdm(dataloader, desc="Extracting both 1d and 2d embeddings"):
        waveform = batch["waveform"].to(device)
        spectrogram = batch["spectrogram"].to(device)

        h1, h2 = contrastive_model.get_embeddings(waveform, spectrogram)

        all_embeddings_1d.append(h1.cpu())
        all_embeddings_2d.append(h2.cpu())
        all_digits.append(batch["digit"].cpu())
        all_speakers.append(batch["speaker_id"].cpu())

    embeddings_1d = torch.cat(all_embeddings_1d, dim=0).numpy()
    embeddings_2d = torch.cat(all_embeddings_2d, dim=0).numpy()
    digits = torch.cat(all_digits, dim=0).numpy()
    speakers = torch.cat(all_speakers, dim=0).numpy()

    return embeddings_1d, embeddings_2d, speakers, digits
