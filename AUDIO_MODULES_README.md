# Audio Self-Supervised Learning Modules

## Testing Status: ✅ ALL TESTS PASSED

All modules have been implemented and tested successfully. You can now use them in your homework notebook.

## Module Overview

### 📁 audio_data/
- **transforms.py**: Audio preprocessing
  - `LogMelSpectrogram`: Convert waveform to log mel-spectrogram (64 mels)
  - `ResampleAudio`: Resample from 48kHz to 16kHz
  - `NormalizeAudio`: Peak normalization

- **augmentations.py**: Audio augmentations
  - `AddGaussianNoise`: Add noise with controlled SNR (15-25 dB)
  - `RandomGain`: Random gain adjustment (±3 dB)
  - `SpecAugment`: Time/frequency masking for spectrograms
  - `WaveformAugmentation`: Combined waveform augmentation pipeline

- **dataset.py**: Data loading
  - `AudioMNISTDataset`: Load audio files with proper speaker/digit labels
  - `AudioMNISTDataModule`: PyTorch Lightning data module with stratified splits
    - Test: speakers 41-60 (20 speakers)
    - Train/Val: speakers 1-40 (stratified 80/20 split)

### 🧠 audio_models/
- **encoders.py**: Neural network architectures
  - `Encoder1D`: 4-layer dilated 1D CNN for raw waveforms (→ 512-dim)
  - `Encoder2D`: 4-layer 2D CNN for spectrograms (→ 512-dim)

- **losses.py**: Loss functions
  - `InfoNCELoss`: Contrastive loss with temperature scaling (τ=0.07)

- **supervised.py**: Supervised baseline
  - `SupervisedModel`: Encoder + classifier for digit classification

- **contrastive.py**: Self-supervised learning
  - `MultiFormatContrastiveModel`: Multi-format contrastive learning
    - Contrasts 1D (waveform) vs 2D (spectrogram) embeddings
    - Projection heads: MLP [512→512→256]
  - `FinetuneModel`: Fine-tune contrastive models for classification

### 📊 audio_utils/
- **linear_probe.py**: Evaluation utilities
  - `LinearProbe.extract_features()`: Extract embeddings from models
  - `LinearProbe.train_and_evaluate()`: Train sklearn LogisticRegression
  - `LinearProbe.evaluate_multimodal()`: Evaluate 1D, 2D, and concat features

- **visualization.py**: t-SNE plotting
  - `plot_tsne()`: Create t-SNE plots colored by digit or speaker
  - `generate_all_tsne_plots()`: Generate all 8 plots (4 models × 2 colorings)

## Quick Start Example

```python
# Import modules
from audio_data import AudioMNISTDataModule
from audio_models import SupervisedModel, MultiFormatContrastiveModel
from audio_utils import LinearProbe, generate_all_tsne_plots
import lightning as pl

# ============================================================
# Task 1: Supervised Baseline
# ============================================================

# 1D model (raw waveforms)
datamodule_1d = AudioMNISTDataModule(
    root='AudioMNIST/data',
    batch_size=32,
    mode='supervised_1d',
)

model_1d = SupervisedModel(encoder_type='1d', num_classes=10, max_epochs=50)
trainer_1d = pl.Trainer(max_epochs=50, accelerator='auto')
trainer_1d.fit(model_1d, datamodule_1d)
trainer_1d.test(model_1d, datamodule_1d)

# 2D model (spectrograms)
datamodule_2d = AudioMNISTDataModule(
    root='AudioMNIST/data',
    batch_size=32,
    mode='supervised_2d',
)

model_2d = SupervisedModel(encoder_type='2d', num_classes=10, max_epochs=50)
trainer_2d = pl.Trainer(max_epochs=50, accelerator='auto')
trainer_2d.fit(model_2d, datamodule_2d)
trainer_2d.test(model_2d, datamodule_2d)

# ============================================================
# Task 2: Contrastive Learning with 4 Augmentation Variants
# ============================================================

configs = [
    {'aug_waveform': False, 'aug_spectrogram': False, 'name': 'no_aug'},
    {'aug_waveform': True, 'aug_spectrogram': False, 'name': 'wave_aug'},
    {'aug_waveform': False, 'aug_spectrogram': True, 'name': 'spec_aug'},
    {'aug_waveform': True, 'aug_spectrogram': True, 'name': 'both_aug'},
]

results = []
for config in configs:
    print(f"\n{'='*60}")
    print(f"Training: {config['name']}")
    print(f"{'='*60}")

    # Create data module
    datamodule = AudioMNISTDataModule(
        root='AudioMNIST/data',
        batch_size=64,
        mode='contrastive',
        apply_waveform_aug=config['aug_waveform'],
        apply_spectrogram_aug=config['aug_spectrogram'],
    )

    # Train contrastive model
    model = MultiFormatContrastiveModel(max_epochs=100)
    trainer = pl.Trainer(max_epochs=100, accelerator='auto')
    trainer.fit(model, datamodule)

    # Linear probe evaluation
    probe_results = LinearProbe.evaluate_multimodal(
        model,
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
    )

    results.append({
        'config': config,
        'model': model,
        'results': probe_results,
    })

# Select best model
best_idx = max(range(len(results)),
               key=lambda i: results[i]['results']['concat']['val_acc'])
best_model = results[best_idx]['model']
print(f"\nBest model: {results[best_idx]['config']['name']}")

# ============================================================
# Task 3: t-SNE Visualization
# ============================================================

# Prepare models for visualization
models_dict = {
    'supervised_1d': (model_1d.encoder, '1d'),
    'supervised_2d': (model_2d.encoder, '2d'),
    'contrastive_1d': (best_model.encoder_1d, '1d'),
    'contrastive_2d': (best_model.encoder_2d, '2d'),
}

# Generate all t-SNE plots
datamodule_test = AudioMNISTDataModule(
    root='AudioMNIST/data',
    batch_size=32,
    mode='contrastive',
)
datamodule_test.setup('test')

generate_all_tsne_plots(
    models_dict,
    datamodule_test.test_dataloader(),
    save_dir='tsne_plots',
)

# ============================================================
# Task 4: Voice Biometrics with kNN
# ============================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Extract embeddings from test speakers
from audio_utils.linear_probe import LinearProbe

test_loader = datamodule_test.test_dataloader()

# For each model
for model_name, (encoder, enc_type) in models_dict.items():
    print(f"\nEvaluating {model_name} for voice biometrics...")

    # Extract features
    features, digits, speakers = LinearProbe.extract_features(
        encoder, test_loader, encoder_type=enc_type
    )

    # Stratified split
    stratify = list(zip(speakers, digits))
    X_train, X_test, y_train, y_test = train_test_split(
        features, speakers, test_size=0.5, stratify=stratify, random_state=42
    )

    # Grid search for k
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 20]}
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Evaluate
    best_acc = grid_search.score(X_test, y_test)
    print(f"  Best k: {grid_search.best_params_['n_neighbors']}")
    print(f"  Speaker accuracy: {best_acc:.4f}")

# ============================================================
# Task 5: Fine-tuning (Bonus)
# ============================================================

from audio_models.contrastive import FinetuneModel

# Fine-tune on digit classification
finetune_model = FinetuneModel(
    contrastive_model=best_model,
    encoder_type='concat',  # Use both encoders
    freeze_encoder=False,   # Allow fine-tuning
    max_epochs=50,
)

trainer_ft = pl.Trainer(max_epochs=50, accelerator='auto')
trainer_ft.fit(finetune_model, datamodule)
trainer_ft.test(finetune_model, datamodule)
```

## Data Statistics

- **Total speakers**: 60
- **Digits**: 0-9 (10 classes)
- **Test set**: 20 speakers (41-60), ~10,000 samples
- **Train set**: ~20,000 samples
- **Val set**: ~20,000 samples

## Architecture Details

### Encoder1D (Raw Waveforms)
- Input: 48,000 samples (3 sec @ 16kHz)
- 4 conv1d layers with dilations [1, 2, 4, 8]
- Channels: [64, 128, 256, 512]
- Kernel size: 16, Stride: [4, 2, 2, 2]
- Output: 512-dim embedding

### Encoder2D (Spectrograms)
- Input: 64 × 94 (mels × time)
- 4 conv2d layers
- Channels: [64, 128, 256, 512]
- Kernel size: 3, Stride: 2
- Output: 512-dim embedding

## Training Hyperparameters

### Supervised
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Batch size: 32
- Epochs: 50

### Contrastive
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Batch size: 64
- Epochs: 100
- Temperature: 0.07

## Testing

Run the comprehensive test suite:
```bash
python test_audio_modules.py
```

All tests should pass with output showing successful initialization and forward passes for all components.

## Known Issues

- torchaudio warnings about deprecated parameters (safe to ignore)
- Lightning warnings about self.trainer (expected when testing without Trainer)

## Next Steps

1. Open `homework-week12.ipynb`
2. Import the modules as shown in the examples above
3. Run training for each task
4. Document your experiments and analysis in the notebook

Good luck with your homework! 🎵🎓
