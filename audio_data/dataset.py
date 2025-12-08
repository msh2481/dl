import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from sklearn.model_selection import train_test_split

from .transforms import LogMelSpectrogram, ResampleAudio, NormalizeAudio
from .augmentations import WaveformAugmentation, SpecAugment


class AudioMNISTDataset(Dataset):
    def __init__(
        self,
        root: str,
        speaker_ids: List[int],
        target_sample_rate: int = 16000,
        target_length: int = 48000,
        apply_waveform_aug: bool = False,
        apply_spectrogram_aug: bool = False,
        mode: str = "contrastive",
        ratio: float = 1.0,
    ):
        super().__init__()
        self.root = Path(root)
        self.speaker_ids = speaker_ids
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.mode = mode

        self.resample = ResampleAudio(
            orig_freq=48000, new_freq=target_sample_rate
        )
        self.normalize = NormalizeAudio(method="peak")
        self.mel_transform = LogMelSpectrogram(
            sample_rate=target_sample_rate, n_mels=64
        )

        self.waveform_aug = WaveformAugmentation(apply=apply_waveform_aug)
        self.spectrogram_aug = SpecAugment(
            p=0.8 if apply_spectrogram_aug else 0.0
        )

        self.files = []
        self._load_file_paths(ratio=ratio)

    def _load_file_paths(self, ratio: float = 1.0):
        for speaker_id in self.speaker_ids:
            speaker_dir = self.root / f"{speaker_id:02d}"
            if not speaker_dir.exists():
                continue

            for audio_file in speaker_dir.glob("*.wav"):
                parts = audio_file.stem.split("_")
                if len(parts) >= 2:
                    digit = int(parts[0])
                    self.files.append(
                        {
                            "path": audio_file,
                            "speaker_id": speaker_id,
                            "digit": digit,
                        }
                    )

        if ratio < 1.0:
            import random

            random.seed(42)
            n_samples = int(len(self.files) * ratio)
            self.files = random.sample(self.files, n_samples)

    def _process_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self.resample(waveform)

        waveform = self.normalize(waveform)

        current_length = waveform.shape[-1]
        if current_length < self.target_length:
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > self.target_length:
            start = (current_length - self.target_length) // 2
            waveform = waveform[..., start : start + self.target_length]

        return waveform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        file_info = self.files[idx]
        waveform, orig_sr = torchaudio.load(file_info["path"])
        waveform = self._process_waveform(waveform)
        waveform_aug = self.waveform_aug(waveform)
        spectrogram = self.mel_transform(waveform_aug.squeeze(0))
        spectrogram = self.spectrogram_aug(spectrogram)

        if self.mode == "contrastive":
            return {
                "waveform": waveform_aug.squeeze(0),
                "spectrogram": spectrogram,
                "speaker_id": file_info["speaker_id"],
                "digit": file_info["digit"],
            }
        elif self.mode == "supervised_1d":
            return {
                "waveform": waveform_aug.squeeze(0),
                "digit": file_info["digit"],
                "speaker_id": file_info["speaker_id"],
            }
        elif self.mode == "supervised_2d":
            return {
                "spectrogram": spectrogram,
                "digit": file_info["digit"],
                "speaker_id": file_info["speaker_id"],
            }
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class AudioMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "AudioMNIST/data",
        batch_size: int = 32,
        num_workers: int = 4,
        mode: str = "contrastive",
        apply_waveform_aug: bool = False,
        apply_spectrogram_aug: bool = False,
        test_speaker_start: int = 41,
        ratio: float = 1.0,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.apply_waveform_aug = apply_waveform_aug
        self.apply_spectrogram_aug = apply_spectrogram_aug
        self.test_speaker_start = test_speaker_start
        self.ratio = ratio

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        test_speakers = list(range(self.test_speaker_start, 61))

        train_val_speakers = list(range(1, self.test_speaker_start))

        temp_dataset = AudioMNISTDataset(
            root=self.root,
            speaker_ids=train_val_speakers,
            mode=self.mode,
            ratio=self.ratio,
        )

        stratify_labels = [
            (item["speaker_id"], item["digit"]) for item in temp_dataset.files
        ]
        indices = list(range(len(temp_dataset.files)))

        try:
            train_indices, val_indices = train_test_split(
                indices,
                test_size=0.2,
                stratify=stratify_labels,
                random_state=42,
            )
        except ValueError:
            train_indices, val_indices = train_test_split(
                indices,
                test_size=0.2,
                stratify=None,
                random_state=42,
            )

        train_speakers_set = set()
        val_speakers_set = set()

        for idx in train_indices:
            train_speakers_set.add(temp_dataset.files[idx]["speaker_id"])

        for idx in val_indices:
            val_speakers_set.add(temp_dataset.files[idx]["speaker_id"])

        train_speakers = sorted(list(train_speakers_set))
        val_speakers = sorted(list(val_speakers_set))

        if stage == "fit" or stage is None:
            self.train_dataset = AudioMNISTDataset(
                root=self.root,
                speaker_ids=train_speakers,
                apply_waveform_aug=self.apply_waveform_aug,
                apply_spectrogram_aug=self.apply_spectrogram_aug,
                mode=self.mode,
                ratio=self.ratio,
            )

            self.val_dataset = AudioMNISTDataset(
                root=self.root,
                speaker_ids=val_speakers,
                apply_waveform_aug=False,
                apply_spectrogram_aug=False,
                mode=self.mode,
                ratio=self.ratio,
            )

        if stage == "test" or stage is None:
            self.test_dataset = AudioMNISTDataset(
                root=self.root,
                speaker_ids=test_speakers,
                apply_waveform_aug=False,
                apply_spectrogram_aug=False,
                mode=self.mode,
                ratio=self.ratio,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
