"""Encoder architectures for audio data."""

import torch
import torch.nn as nn


class Encoder1D(nn.Module):
    """1D CNN encoder for raw audio waveforms.

    Uses dilated convolutions with large kernels for efficient temporal modeling.

    Args:
        input_channels: Number of input channels (default: 1)
        embedding_dim: Output embedding dimension (default: 512)
        channels: List of channel dimensions for each conv layer (default: [64, 128, 256, 512])
        kernel_size: Kernel size for convolutions (default: 16)
        stride: Stride for downsampling (default: 4)
    """

    def __init__(
        self,
        input_channels: int = 1,
        embedding_dim: int = 512,
        channels: list = None,
        kernel_size: int = 16,
        stride: int = 4,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512]

        layers = []
        in_channels = input_channels

        # Build conv blocks with increasing dilation
        dilations = [1, 2, 4, 8]
        for i, out_channels in enumerate(channels):
            dilation = dilations[i] if i < len(dilations) else 1

            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 2,  # First layer has stride=4, others stride=2
                    padding=(kernel_size * dilation) // 2,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Projection to embedding dimension if needed
        self.projection = nn.Linear(channels[-1], embedding_dim) if channels[-1] != embedding_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input waveform tensor of shape (batch, time) or (batch, 1, time)

        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        # Ensure input has channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)

        # Apply convolutions
        x = self.encoder(x)  # (batch, channels, time')

        # Global pooling
        x = self.pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)

        # Project to embedding dimension
        x = self.projection(x)  # (batch, embedding_dim)

        return x


class Encoder2D(nn.Module):
    """2D CNN encoder for spectrograms.

    Uses standard 2D convolutions with batch normalization.

    Args:
        input_channels: Number of input channels (default: 1)
        embedding_dim: Output embedding dimension (default: 512)
        channels: List of channel dimensions for each conv block (default: [64, 128, 256, 512])
        kernel_size: Kernel size for convolutions (default: 3)
    """

    def __init__(
        self,
        input_channels: int = 1,
        embedding_dim: int = 512,
        channels: list = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512]

        layers = []
        in_channels = input_channels

        # Build conv blocks
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection to embedding dimension if needed
        self.projection = nn.Linear(channels[-1], embedding_dim) if channels[-1] != embedding_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input spectrogram tensor of shape (batch, freq, time) or (batch, 1, freq, time)

        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        # Ensure input has channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, freq, time)

        # Apply convolutions
        x = self.encoder(x)  # (batch, channels, freq', time')

        # Adaptive pooling
        x = self.pool(x)  # (batch, channels, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch, channels)

        # Project to embedding dimension
        x = self.projection(x)  # (batch, embedding_dim)

        return x


def test_encoders():
    """Test encoder architectures with dummy data."""
    # Test 1D encoder
    encoder_1d = Encoder1D(embedding_dim=512)
    waveform = torch.randn(4, 48000)  # Batch of 4, 3 seconds @ 16kHz
    embedding_1d = encoder_1d(waveform)
    print(f"1D Encoder - Input: {waveform.shape}, Output: {embedding_1d.shape}")
    assert embedding_1d.shape == (4, 512)

    # Test 2D encoder
    encoder_2d = Encoder2D(embedding_dim=512)
    spectrogram = torch.randn(4, 64, 94)  # Batch of 4, 64 mels, ~94 time frames
    embedding_2d = encoder_2d(spectrogram)
    print(f"2D Encoder - Input: {spectrogram.shape}, Output: {embedding_2d.shape}")
    assert embedding_2d.shape == (4, 512)

    print("All encoder tests passed!")


if __name__ == '__main__':
    test_encoders()
