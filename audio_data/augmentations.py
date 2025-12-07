import torch
import torch.nn as nn
import torch.nn.functional as F


class AddGaussianNoise(nn.Module):
    def __init__(self, snr_db_range: tuple = (15, 25), p: float = 0.5):
        super().__init__()
        self.snr_db_min, self.snr_db_max = snr_db_range
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.training and torch.rand(1).item() < self.p:
            snr_db = (
                torch.rand(1).item() * (self.snr_db_max - self.snr_db_min)
                + self.snr_db_min
            )

            signal_power = torch.mean(waveform**2)

            snr_linear = 10 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear

            noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
            waveform = waveform + noise

        return waveform


class RandomGain(nn.Module):
    def __init__(self, gain_db_range: tuple = (-6, 6), p: float = 0.5):
        super().__init__()
        self.gain_db_min, self.gain_db_max = gain_db_range
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.training and torch.rand(1).item() < self.p:
            gain_db = (
                torch.rand(1).item() * (self.gain_db_max - self.gain_db_min)
                + self.gain_db_min
            )
            gain_linear = 10 ** (gain_db / 20.0)

            waveform = waveform * gain_linear
            waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform


class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        p: float = 0.5,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() >= self.p:
            return spectrogram

        spec = spectrogram.clone()
        *batch_dims, freq_dim, time_dim = spec.shape

        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param, (1,)).item()
            f0 = torch.randint(0, max(1, freq_dim - f), (1,)).item()
            spec[..., f0 : f0 + f, :] = 0

        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param, (1,)).item()
            t0 = torch.randint(0, max(1, time_dim - t), (1,)).item()
            spec[..., :, t0 : t0 + t] = 0

        return spec


class WaveformAugmentation(nn.Module):
    def __init__(
        self,
        apply: bool = True,
        snr_db_range: tuple = (15, 25),
        gain_db_range: tuple = (-3, 3),
    ):
        super().__init__()
        self.apply = apply
        if apply:
            self.noise = AddGaussianNoise(snr_db_range=snr_db_range, p=0.8)
            self.gain = RandomGain(gain_db_range=gain_db_range, p=0.8)
        else:
            self.noise = nn.Identity()
            self.gain = nn.Identity()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.apply:
            waveform = self.noise(waveform)
            waveform = self.gain(waveform)
        return waveform
