"""Audio preprocessing and feature extraction transforms."""

import torch
import torch.nn as nn
import torchaudio.transforms as T


class LogMelSpectrogram(nn.Module):
    """Convert waveform to log mel-spectrogram.

    Args:
        sample_rate: Audio sample rate (default: 16000)
        n_mels: Number of mel filterbanks (default: 64)
        n_fft: Size of FFT (default: 1024)
        hop_length: Number of samples between successive frames (default: 512)
        eps: Small value to avoid log(0) (default: 1e-8)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to log mel-spectrogram.

        Args:
            waveform: Input waveform tensor of shape (batch, 1, time) or (1, time)

        Returns:
            Log mel-spectrogram of shape (batch, n_mels, time_frames) or (n_mels, time_frames)
        """
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = (mel_spec + self.eps).log()
        return log_mel_spec


class ResampleAudio(nn.Module):
    """Resample audio from original frequency to target frequency.

    Args:
        orig_freq: Original sample rate (default: 48000)
        new_freq: Target sample rate (default: 16000)
    """

    def __init__(self, orig_freq: int = 48000, new_freq: int = 16000):
        super().__init__()
        self.resampler = T.Resample(orig_freq=orig_freq, new_freq=new_freq)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Resample waveform.

        Args:
            waveform: Input waveform tensor

        Returns:
            Resampled waveform
        """
        return self.resampler(waveform)


class NormalizeAudio(nn.Module):
    """Normalize audio waveform to [-1, 1] range.

    Args:
        method: Normalization method ('peak' or 'rms')
        target_level: Target RMS level in dB (only used if method='rms')
    """

    def __init__(self, method: str = 'peak', target_level: float = -20.0):
        super().__init__()
        self.method = method
        self.target_level = target_level

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize waveform.

        Args:
            waveform: Input waveform tensor

        Returns:
            Normalized waveform
        """
        if self.method == 'peak':
            # Peak normalization
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak
        elif self.method == 'rms':
            # RMS normalization
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms > 0:
                target_rms = 10 ** (self.target_level / 20.0)
                waveform = waveform * (target_rms / rms)
                # Clip to [-1, 1]
                waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform
