"""Audio augmentation functions for waveforms and spectrograms."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddGaussianNoise(nn.Module):
    """Add Gaussian noise to waveform with specified SNR.

    Args:
        snr_db_range: Range of SNR in dB (min, max) (default: (15, 25))
        p: Probability of applying augmentation (default: 0.5)
    """

    def __init__(self, snr_db_range: tuple = (15, 25), p: float = 0.5):
        super().__init__()
        self.snr_db_min, self.snr_db_max = snr_db_range
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to waveform.

        Args:
            waveform: Input waveform tensor of shape (..., time)

        Returns:
            Noisy waveform of same shape
        """
        if self.training and torch.rand(1).item() < self.p:
            # Sample random SNR
            snr_db = torch.rand(1).item() * (self.snr_db_max - self.snr_db_min) + self.snr_db_min

            # Calculate signal power
            signal_power = torch.mean(waveform ** 2)

            # Calculate noise power from SNR
            snr_linear = 10 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear

            # Generate and add noise
            noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
            waveform = waveform + noise

        return waveform


class RandomGain(nn.Module):
    """Apply random gain to waveform.

    Args:
        gain_db_range: Range of gain in dB (min, max) (default: (-6, 6))
        p: Probability of applying augmentation (default: 0.5)
    """

    def __init__(self, gain_db_range: tuple = (-6, 6), p: float = 0.5):
        super().__init__()
        self.gain_db_min, self.gain_db_max = gain_db_range
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random gain to waveform.

        Args:
            waveform: Input waveform tensor of shape (..., time)

        Returns:
            Waveform with applied gain
        """
        if self.training and torch.rand(1).item() < self.p:
            # Sample random gain
            gain_db = torch.rand(1).item() * (self.gain_db_max - self.gain_db_min) + self.gain_db_min
            gain_linear = 10 ** (gain_db / 20.0)

            # Apply gain and clip
            waveform = waveform * gain_linear
            waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform


class SpecAugment(nn.Module):
    """Apply SpecAugment to spectrograms (Park et al. 2019).

    Masks random time and frequency regions in the spectrogram.

    Args:
        freq_mask_param: Maximum frequency mask width (default: 15)
        time_mask_param: Maximum time mask width (default: 35)
        num_freq_masks: Number of frequency masks to apply (default: 1)
        num_time_masks: Number of time masks to apply (default: 1)
        p: Probability of applying augmentation (default: 0.5)
    """

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
        """Apply SpecAugment to spectrogram.

        Args:
            spectrogram: Input spectrogram tensor of shape (..., freq, time)

        Returns:
            Augmented spectrogram of same shape
        """
        if not self.training or torch.rand(1).item() >= self.p:
            return spectrogram

        spec = spectrogram.clone()
        *batch_dims, freq_dim, time_dim = spec.shape

        # Apply frequency masks
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param, (1,)).item()
            f0 = torch.randint(0, max(1, freq_dim - f), (1,)).item()
            spec[..., f0:f0 + f, :] = 0

        # Apply time masks
        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param, (1,)).item()
            t0 = torch.randint(0, max(1, time_dim - t), (1,)).item()
            spec[..., :, t0:t0 + t] = 0

        return spec


class WaveformAugmentation(nn.Module):
    """Combined waveform augmentation pipeline.

    Chains AddGaussianNoise and RandomGain.

    Args:
        apply: Whether to apply augmentations (default: True)
        snr_db_range: SNR range for noise (default: (15, 25))
        gain_db_range: Gain range (default: (-3, 3))
    """

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
        """Apply waveform augmentations.

        Args:
            waveform: Input waveform tensor

        Returns:
            Augmented waveform
        """
        if self.apply:
            waveform = self.noise(waveform)
            waveform = self.gain(waveform)
        return waveform
