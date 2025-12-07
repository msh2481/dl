"""Contrastive loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE loss (NT-Xent) for contrastive learning.

    This loss contrasts positive pairs against negative pairs in a batch.
    Based on the SimCLR paper (Chen et al. 2020).

    Args:
        temperature: Temperature parameter for scaling similarities (default: 0.07)
        reduction: Loss reduction method ('mean' or 'sum') (default: 'mean')
    """

    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            z1: Embeddings from first view (batch_size, embedding_dim)
            z2: Embeddings from second view (batch_size, embedding_dim)

        Returns:
            Scalar loss value
        """
        batch_size = z1.shape[0]
        device = z1.device

        # Normalize embeddings to unit sphere
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute similarity matrix
        # Shape: (batch_size, batch_size)
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature

        # Create labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=device)

        # Compute loss (symmetric)
        # For each sample in z1, the positive is the corresponding sample in z2
        loss_1to2 = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction)

        # For each sample in z2, the positive is the corresponding sample in z1
        loss_2to1 = F.cross_entropy(similarity_matrix.T, labels, reduction=self.reduction)

        # Average both directions
        loss = (loss_1to2 + loss_2to1) / 2.0

        return loss


def test_infonce_loss():
    """Test InfoNCE loss implementation."""
    batch_size = 8
    embedding_dim = 128

    # Create dummy embeddings
    z1 = torch.randn(batch_size, embedding_dim)
    z2 = torch.randn(batch_size, embedding_dim)

    # Test loss
    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(z1, z2)

    print(f"InfoNCE Loss: {loss.item():.4f}")
    print(f"Loss shape: {loss.shape}")
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"

    # Test with identical embeddings (should give low loss)
    z_same = torch.randn(batch_size, embedding_dim)
    loss_same = loss_fn(z_same, z_same.clone())
    print(f"Loss with identical embeddings: {loss_same.item():.4f}")

    # Test with similar embeddings (should give lower loss than random)
    z_a = torch.randn(batch_size, embedding_dim)
    z_b = z_a + 0.1 * torch.randn(batch_size, embedding_dim)  # Add small noise
    loss_similar = loss_fn(z_a, z_b)
    print(f"Loss with similar embeddings: {loss_similar.item():.4f}")

    print("All InfoNCE loss tests passed!")


if __name__ == '__main__':
    test_infonce_loss()
