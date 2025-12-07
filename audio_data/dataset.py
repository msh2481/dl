"""AudioMNIST dataset and data module."""

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
    """AudioMNIST dataset for loading audio files.

    Args:
        root: Root directory containing speaker subdirectories
        speaker_ids: List of speaker IDs to include in this dataset
        target_sample_rate: Target sample rate after resampling (default: 16000)
        target_length: Target length in samples for padding/cropping (default: 48000 = 3 seconds @ 16kHz)
        apply_waveform_aug: Whether to apply waveform augmentations (default: False)
        apply_spectrogram_aug: Whether to apply spectrogram augmentations (default: False)
        mode: Dataset mode - 'contrastive' returns (waveform, spectrogram) or 'supervised' returns only one
    """

    def __init__(
        self,
        root: str,
        speaker_ids: List[int],
        target_sample_rate: int = 16000,
        target_length: int = 48000,
        apply_waveform_aug: bool = False,
        apply_spectrogram_aug: bool = False,
        mode: str = 'contrastive',
    ):
        super().__init__()
        self.root = Path(root)
        self.speaker_ids = speaker_ids
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.mode = mode

        # Transforms
        self.resample = ResampleAudio(orig_freq=48000, new_freq=target_sample_rate)
        self.normalize = NormalizeAudio(method='peak')
        self.mel_transform = LogMelSpectrogram(sample_rate=target_sample_rate, n_mels=64)

        # Augmentations
        self.waveform_aug = WaveformAugmentation(apply=apply_waveform_aug)
        self.spectrogram_aug = SpecAugment(p=0.8 if apply_spectrogram_aug else 0.0)

        # Load file paths
        self.files = []
        self._load_file_paths()

    def _load_file_paths(self):
        """Load all audio file paths for specified speakers."""
        for speaker_id in self.speaker_ids:
            # Speaker directories are zero-padded (01, 02, ..., 60)
            speaker_dir = self.root / f"{speaker_id:02d}"
            if not speaker_dir.exists():
                continue

            for audio_file in speaker_dir.glob('*.wav'):
                # Parse filename: {digit}_{speaker}_{repetition}.wav
                parts = audio_file.stem.split('_')
                if len(parts) >= 2:
                    digit = int(parts[0])
                    self.files.append({
                        'path': audio_file,
                        'speaker_id': speaker_id,
                        'digit': digit,
                    })

    def _process_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Process waveform: resample, normalize, pad/crop."""
        # Resample
        waveform = self.resample(waveform)

        # Normalize
        waveform = self.normalize(waveform)

        # Pad or crop to target length
        current_length = waveform.shape[-1]
        if current_length < self.target_length:
            # Pad
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > self.target_length:
            # Crop (take center)
            start = (current_length - self.target_length) // 2
            waveform = waveform[..., start:start + self.target_length]

        return waveform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        """Get audio sample.

        Returns:
            dict with keys:
                - 'waveform': Raw waveform tensor (1, target_length)
                - 'spectrogram': Log mel-spectrogram (64, time_frames)
                - 'speaker_id': Speaker ID
                - 'digit': Digit label (0-9)
        """
        file_info = self.files[idx]

        # Load audio
        waveform, orig_sr = torchaudio.load(file_info['path'])

        # Process waveform
        waveform = self._process_waveform(waveform)

        # Apply waveform augmentation
        waveform_aug = self.waveform_aug(waveform)

        # Convert to spectrogram
        spectrogram = self.mel_transform(waveform_aug.squeeze(0))

        # Apply spectrogram augmentation
        spectrogram = self.spectrogram_aug(spectrogram)

        # Prepare output based on mode
        if self.mode == 'contrastive':
            return {
                'waveform': waveform_aug.squeeze(0),  # (target_length,)
                'spectrogram': spectrogram,  # (n_mels, time_frames)
                'speaker_id': file_info['speaker_id'],
                'digit': file_info['digit'],
            }
        elif self.mode == 'supervised_1d':
            return {
                'waveform': waveform_aug.squeeze(0),
                'digit': file_info['digit'],
                'speaker_id': file_info['speaker_id'],
            }
        elif self.mode == 'supervised_2d':
            return {
                'spectrogram': spectrogram,
                'digit': file_info['digit'],
                'speaker_id': file_info['speaker_id'],
            }
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class AudioMNISTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for AudioMNIST.

    Args:
        root: Root directory containing AudioMNIST data
        batch_size: Batch size for dataloaders (default: 32)
        num_workers: Number of workers for dataloaders (default: 4)
        mode: Dataset mode ('contrastive', 'supervised_1d', 'supervised_2d')
        apply_waveform_aug: Apply waveform augmentations (default: False)
        apply_spectrogram_aug: Apply spectrogram augmentations (default: False)
        test_speaker_start: First speaker ID for test set (default: 41)
    """

    def __init__(
        self,
        root: str = 'AudioMNIST/data',
        batch_size: int = 32,
        num_workers: int = 4,
        mode: str = 'contrastive',
        apply_waveform_aug: bool = False,
        apply_spectrogram_aug: bool = False,
        test_speaker_start: int = 41,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.apply_waveform_aug = apply_waveform_aug
        self.apply_spectrogram_aug = apply_spectrogram_aug
        self.test_speaker_start = test_speaker_start

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        # Define splits
        # Test: speakers 41-60 (20 speakers, 1/3)
        test_speakers = list(range(self.test_speaker_start, 61))

        # Train/val: speakers 1-40 (40 speakers)
        train_val_speakers = list(range(1, self.test_speaker_start))

        # Stratified split for train/val
        # Load a temporary dataset to get digit distribution
        temp_dataset = AudioMNISTDataset(
            root=self.root,
            speaker_ids=train_val_speakers,
            mode=self.mode,
        )

        # Create stratification labels (speaker_id, digit)
        stratify_labels = [(item['speaker_id'], item['digit']) for item in temp_dataset.files]
        indices = list(range(len(temp_dataset.files)))

        # Split 80/20 for train/val with stratification
        train_indices, val_indices = train_test_split(
            indices,
            test_size=0.2,
            stratify=stratify_labels,
            random_state=42,
        )

        # Get unique speaker IDs for each split
        train_speakers_set = set()
        val_speakers_set = set()

        for idx in train_indices:
            train_speakers_set.add(temp_dataset.files[idx]['speaker_id'])

        for idx in val_indices:
            val_speakers_set.add(temp_dataset.files[idx]['speaker_id'])

        train_speakers = sorted(list(train_speakers_set))
        val_speakers = sorted(list(val_speakers_set))

        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = AudioMNISTDataset(
                root=self.root,
                speaker_ids=train_speakers,
                apply_waveform_aug=self.apply_waveform_aug,
                apply_spectrogram_aug=self.apply_spectrogram_aug,
                mode=self.mode,
            )

            self.val_dataset = AudioMNISTDataset(
                root=self.root,
                speaker_ids=val_speakers,
                apply_waveform_aug=False,  # No augmentation for validation
                apply_spectrogram_aug=False,
                mode=self.mode,
            )

        if stage == 'test' or stage is None:
            self.test_dataset = AudioMNISTDataset(
                root=self.root,
                speaker_ids=test_speakers,
                apply_waveform_aug=False,  # No augmentation for test
                apply_spectrogram_aug=False,
                mode=self.mode,
            )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
