"""Visualization utilities for embeddings."""

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
    label_type='digit',
    perplexity=30,
    n_iter=1000,
    figsize=(10, 8),
):
    """Plot t-SNE visualization of embeddings.

    Args:
        embeddings: Embedding array of shape (n_samples, embedding_dim)
        labels: Label array of shape (n_samples,)
        title: Plot title
        save_path: Path to save figure
        label_type: Type of labels ('digit' or 'speaker')
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
        figsize: Figure size
    """
    print(f"Computing t-SNE for {title}...")

    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        verbose=0,
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colormap and number of classes
    if label_type == 'digit':
        n_classes = 10
        cmap = plt.cm.tab10
        label_name = 'Digit'
    else:  # speaker
        n_classes = len(np.unique(labels))
        cmap = plt.cm.tab20 if n_classes <= 20 else plt.cm.viridis
        label_name = 'Speaker'

    # Plot each class with different color
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

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(title)

    # Add legend
    if label_type == 'digit':
        ax.legend(title=label_name, loc='best', ncol=2)
    else:
        # For speakers, legend might be too large, so optionally skip or use colorbar
        if n_classes <= 20:
            ax.legend(title=label_name, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
        else:
            # Use colorbar instead
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_classes))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(label_name)

    plt.tight_layout()

    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved t-SNE plot to {save_path}")


@torch.no_grad()
def extract_embeddings(encoder, dataloader, device='cuda', encoder_type='1d'):
    """Extract embeddings from encoder for entire dataset.

    Args:
        encoder: Encoder model
        dataloader: DataLoader
        device: Device to use
        encoder_type: Type of encoder ('1d' or '2d')

    Returns:
        Tuple of (embeddings, digit_labels, speaker_ids) as numpy arrays
    """
    encoder.eval()
    encoder = encoder.to(device)

    all_embeddings = []
    all_digits = []
    all_speakers = []

    for batch in tqdm(dataloader, desc=f"Extracting embeddings"):
        if encoder_type == '1d':
            inputs = batch['waveform'].to(device)
        else:
            inputs = batch['spectrogram'].to(device)

        embeddings = encoder(inputs)

        all_embeddings.append(embeddings.cpu())
        all_digits.append(batch['digit'].cpu())
        all_speakers.append(batch['speaker_id'].cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    digits = torch.cat(all_digits, dim=0).numpy()
    speakers = torch.cat(all_speakers, dim=0).numpy()

    return embeddings, digits, speakers


def generate_all_tsne_plots(
    models_dict,
    dataloader,
    save_dir='tsne_plots',
    device='cuda',
    perplexity=30,
):
    """Generate t-SNE plots for all models.

    Args:
        models_dict: Dictionary with model names as keys and (encoder, encoder_type) as values
            Example: {
                'supervised_1d': (encoder_1d, '1d'),
                'supervised_2d': (encoder_2d, '2d'),
                'contrastive_1d': (encoder_1d, '1d'),
                'contrastive_2d': (encoder_2d, '2d'),
            }
        dataloader: Test dataloader
        save_dir: Directory to save plots
        device: Device to use
        perplexity: t-SNE perplexity parameter
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for model_name, (encoder, encoder_type) in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Generating t-SNE plots for {model_name}")
        print(f"{'='*60}")

        # Extract embeddings
        embeddings, digits, speakers = extract_embeddings(
            encoder, dataloader, device, encoder_type
        )

        # Generate plot colored by digit
        plot_tsne(
            embeddings=embeddings,
            labels=digits,
            title=f'{model_name} - Colored by Digit',
            save_path=os.path.join(save_dir, f'{model_name}_by_digit.png'),
            label_type='digit',
            perplexity=perplexity,
        )

        # Generate plot colored by speaker
        plot_tsne(
            embeddings=embeddings,
            labels=speakers,
            title=f'{model_name} - Colored by Speaker',
            save_path=os.path.join(save_dir, f'{model_name}_by_speaker.png'),
            label_type='speaker',
            perplexity=perplexity,
        )

    print(f"\n{'='*60}")
    print(f"All t-SNE plots saved to {save_dir}")
    print(f"{'='*60}")


@torch.no_grad()
def extract_contrastive_embeddings(
    contrastive_model,
    dataloader,
    device='cuda',
    encoder_type='1d',
):
    """Extract embeddings from contrastive model.

    Args:
        contrastive_model: Contrastive model with encoder_1d and encoder_2d
        dataloader: DataLoader
        device: Device to use
        encoder_type: Which encoder to use ('1d' or '2d')

    Returns:
        Tuple of (embeddings, digit_labels, speaker_ids) as numpy arrays
    """
    contrastive_model.eval()
    contrastive_model = contrastive_model.to(device)

    all_embeddings = []
    all_digits = []
    all_speakers = []

    for batch in tqdm(dataloader, desc=f"Extracting {encoder_type} embeddings"):
        waveform = batch['waveform'].to(device)
        spectrogram = batch['spectrogram'].to(device)

        # Get embeddings
        h1, h2 = contrastive_model.get_embeddings(waveform, spectrogram)

        if encoder_type == '1d':
            embeddings = h1
        else:
            embeddings = h2

        all_embeddings.append(embeddings.cpu())
        all_digits.append(batch['digit'].cpu())
        all_speakers.append(batch['speaker_id'].cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    digits = torch.cat(all_digits, dim=0).numpy()
    speakers = torch.cat(all_speakers, dim=0).numpy()

    return embeddings, digits, speakers
