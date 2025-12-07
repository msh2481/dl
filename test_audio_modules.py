"""Comprehensive test script for all audio modules."""

import torch
from audio_data import (
    LogMelSpectrogram, ResampleAudio, NormalizeAudio,
    AddGaussianNoise, RandomGain, SpecAugment,
    AudioMNISTDataset, AudioMNISTDataModule
)
from audio_models import (
    Encoder1D, Encoder2D, InfoNCELoss,
    MultiFormatContrastiveModel, SupervisedModel
)
from audio_utils import LinearProbe


def test_transforms():
    """Test audio transforms."""
    print("\n" + "="*60)
    print("Testing Audio Transforms")
    print("="*60)

    waveform = torch.randn(2, 48000)

    # LogMelSpectrogram
    mel_transform = LogMelSpectrogram(sample_rate=16000, n_mels=64)
    mel_spec = mel_transform(waveform)
    print(f"✓ LogMelSpectrogram: {waveform.shape} -> {mel_spec.shape}")
    assert mel_spec.shape[1] == 64, "Should have 64 mel bins"

    # ResampleAudio
    resample = ResampleAudio(orig_freq=48000, new_freq=16000)
    resampled = resample(waveform)
    print(f"✓ ResampleAudio: {waveform.shape} -> {resampled.shape}")
    assert resampled.shape[-1] == 16000, "Should resample to 16kHz"

    # NormalizeAudio
    normalize = NormalizeAudio(method='peak')
    normalized = normalize(waveform)
    print(f"✓ NormalizeAudio: peak normalized")
    assert normalized.abs().max() <= 1.0, "Should be normalized"


def test_augmentations():
    """Test audio augmentations."""
    print("\n" + "="*60)
    print("Testing Audio Augmentations")
    print("="*60)

    # AddGaussianNoise
    waveform = torch.randn(2, 48000)
    noise_aug = AddGaussianNoise(snr_db_range=(15, 25), p=1.0)
    noise_aug.train()
    noisy = noise_aug(waveform)
    print(f"✓ AddGaussianNoise: {waveform.shape} -> {noisy.shape}")
    assert waveform.shape == noisy.shape

    # RandomGain
    gain_aug = RandomGain(gain_db_range=(-3, 3), p=1.0)
    gain_aug.train()
    gained = gain_aug(waveform)
    print(f"✓ RandomGain: {waveform.shape} -> {gained.shape}")
    assert waveform.shape == gained.shape

    # SpecAugment
    spectrogram = torch.randn(2, 64, 94)
    spec_aug = SpecAugment(freq_mask_param=10, time_mask_param=20, p=1.0)
    spec_aug.train()
    augmented = spec_aug(spectrogram)
    print(f"✓ SpecAugment: {spectrogram.shape} -> {augmented.shape}")
    assert spectrogram.shape == augmented.shape


def test_encoders():
    """Test encoder architectures."""
    print("\n" + "="*60)
    print("Testing Encoder Architectures")
    print("="*60)

    # Encoder1D
    encoder_1d = Encoder1D(embedding_dim=512)
    waveform = torch.randn(4, 48000)
    embedding_1d = encoder_1d(waveform)
    print(f"✓ Encoder1D: {waveform.shape} -> {embedding_1d.shape}")
    assert embedding_1d.shape == (4, 512)

    # Encoder2D
    encoder_2d = Encoder2D(embedding_dim=512)
    spectrogram = torch.randn(4, 64, 94)
    embedding_2d = encoder_2d(spectrogram)
    print(f"✓ Encoder2D: {spectrogram.shape} -> {embedding_2d.shape}")
    assert embedding_2d.shape == (4, 512)


def test_loss():
    """Test contrastive loss."""
    print("\n" + "="*60)
    print("Testing InfoNCE Loss")
    print("="*60)

    loss_fn = InfoNCELoss(temperature=0.07)
    z1 = torch.randn(8, 256)
    z2 = torch.randn(8, 256)
    loss = loss_fn(z1, z2)
    print(f"✓ InfoNCE Loss: {loss.item():.4f}")
    assert loss.dim() == 0, "Loss should be scalar"


def test_dataset():
    """Test AudioMNIST dataset."""
    print("\n" + "="*60)
    print("Testing AudioMNIST Dataset")
    print("="*60)

    # Test contrastive mode
    dataset = AudioMNISTDataset(
        root='AudioMNIST/data',
        speaker_ids=[1, 2, 3],
        mode='contrastive',
    )
    print(f"✓ Dataset created with {len(dataset)} samples")

    sample = dataset[0]
    print(f"✓ Sample keys: {list(sample.keys())}")
    print(f"  - waveform: {sample['waveform'].shape}")
    print(f"  - spectrogram: {sample['spectrogram'].shape}")
    print(f"  - speaker_id: {sample['speaker_id']}, digit: {sample['digit']}")


def test_datamodule():
    """Test AudioMNIST DataModule."""
    print("\n" + "="*60)
    print("Testing AudioMNIST DataModule")
    print("="*60)

    datamodule = AudioMNISTDataModule(
        root='AudioMNIST/data',
        batch_size=4,
        num_workers=0,
        mode='contrastive',
    )

    datamodule.setup('fit')
    print(f"✓ Train dataset: {len(datamodule.train_dataset)} samples")
    print(f"✓ Val dataset: {len(datamodule.val_dataset)} samples")

    datamodule.setup('test')
    print(f"✓ Test dataset: {len(datamodule.test_dataset)} samples")

    # Test batch loading
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    print(f"✓ Batch loaded - waveform: {batch['waveform'].shape}, "
          f"spectrogram: {batch['spectrogram'].shape}")


def test_models():
    """Test model training steps."""
    print("\n" + "="*60)
    print("Testing Model Training Steps")
    print("="*60)

    # Test supervised model
    datamodule = AudioMNISTDataModule(
        root='AudioMNIST/data',
        batch_size=4,
        num_workers=0,
        mode='supervised_1d',
    )
    datamodule.setup('fit')

    model = SupervisedModel(encoder_type='1d', num_classes=10)
    batch = next(iter(datamodule.train_dataloader()))
    model.train()
    loss = model.training_step(batch, 0)
    print(f"✓ Supervised model - Loss: {loss.item():.4f}")

    # Test contrastive model
    datamodule_contrastive = AudioMNISTDataModule(
        root='AudioMNIST/data',
        batch_size=4,
        num_workers=0,
        mode='contrastive',
    )
    datamodule_contrastive.setup('fit')

    contrastive_model = MultiFormatContrastiveModel()
    batch_c = next(iter(datamodule_contrastive.train_dataloader()))
    contrastive_model.train()
    loss_c = contrastive_model.training_step(batch_c, 0)
    print(f"✓ Contrastive model - Loss: {loss_c.item():.4f}")

    # Test embeddings extraction
    contrastive_model.eval()
    with torch.no_grad():
        h1, h2 = contrastive_model.get_embeddings(
            batch_c['waveform'], batch_c['spectrogram']
        )
    print(f"✓ Embeddings extracted - h1: {h1.shape}, h2: {h2.shape}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AUDIO MODULE COMPREHENSIVE TEST SUITE")
    print("="*60)

    try:
        test_transforms()
        test_augmentations()
        test_encoders()
        test_loss()
        test_dataset()
        test_datamodule()
        test_models()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour audio modules are ready to use in the notebook!")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
