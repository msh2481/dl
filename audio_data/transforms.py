import torch
import torch.nn as nn
import torchaudio.transforms as T


class LogMelSpectrogram(nn.Module):
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
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = (mel_spec + self.eps).log()
        return log_mel_spec


class ResampleAudio(nn.Module):
    def __init__(self, orig_freq: int = 48000, new_freq: int = 16000):
        super().__init__()
        self.resampler = T.Resample(orig_freq=orig_freq, new_freq=new_freq)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.resampler(waveform)


class NormalizeAudio(nn.Module):
    def __init__(self, method: str = "peak", target_level: float = -20.0):
        super().__init__()
        self.method = method
        self.target_level = target_level

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.method == "peak":
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak
        elif self.method == "rms":
            rms = torch.sqrt(torch.mean(waveform**2))
            if rms > 0:
                target_rms = 10 ** (self.target_level / 20.0)
                waveform = waveform * (target_rms / rms)
                waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform
