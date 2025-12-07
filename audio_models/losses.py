import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.shape[0]
        device = z1.device

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature

        labels = torch.arange(batch_size, device=device)

        loss_1to2 = F.cross_entropy(
            similarity_matrix, labels, reduction=self.reduction
        )

        loss_2to1 = F.cross_entropy(
            similarity_matrix.T, labels, reduction=self.reduction
        )

        loss = (loss_1to2 + loss_2to1) / 2.0

        return loss


def test_infonce_loss():
    batch_size = 8
    embedding_dim = 128

    z1 = torch.randn(batch_size, embedding_dim)
    z2 = torch.randn(batch_size, embedding_dim)

    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(z1, z2)

    print(f"InfoNCE Loss: {loss.item():.4f}")
    print(f"Loss shape: {loss.shape}")
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"

    z_same = torch.randn(batch_size, embedding_dim)
    loss_same = loss_fn(z_same, z_same.clone())
    print(f"Loss with identical embeddings: {loss_same.item():.4f}")

    z_a = torch.randn(batch_size, embedding_dim)
    z_b = z_a + 0.1 * torch.randn(batch_size, embedding_dim)
    loss_similar = loss_fn(z_a, z_b)
    print(f"Loss with similar embeddings: {loss_similar.item():.4f}")

    print("All InfoNCE loss tests passed!")


if __name__ == "__main__":
    test_infonce_loss()
