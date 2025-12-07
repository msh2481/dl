import torch
import torch.nn as nn


class Encoder1D(nn.Module):
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

        dilations = [1, 2, 4, 8]
        for i, out_channels in enumerate(channels):
            dilation = dilations[i] if i < len(dilations) else 1

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride if i == 0 else 2,
                        padding=(kernel_size * dilation) // 2,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.projection = (
            nn.Linear(channels[-1], embedding_dim)
            if channels[-1] != embedding_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.projection(x)
        return x


class Encoder2D(nn.Module):
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

        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = (
            nn.Linear(channels[-1], embedding_dim)
            if channels[-1] != embedding_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.projection(x)
        return x


def test_encoders():
    encoder_1d = Encoder1D(embedding_dim=512)
    waveform = torch.randn(4, 48000)
    embedding_1d = encoder_1d(waveform)
    print(f"1D Encoder - Input: {waveform.shape}, Output: {embedding_1d.shape}")
    assert embedding_1d.shape == (4, 512)

    encoder_2d = Encoder2D(embedding_dim=512)
    spectrogram = torch.randn(4, 64, 94)
    embedding_2d = encoder_2d(spectrogram)
    print(
        f"2D Encoder - Input: {spectrogram.shape}, Output: {embedding_2d.shape}"
    )
    assert embedding_2d.shape == (4, 512)

    print("All encoder tests passed!")


if __name__ == "__main__":
    test_encoders()
