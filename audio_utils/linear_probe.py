"""Linear probe evaluation utilities."""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class LinearProbe:
    """Linear probe for evaluating learned representations."""

    @staticmethod
    @torch.no_grad()
    def extract_features(
        model,
        dataloader,
        device='cuda',
        encoder_type='1d',
    ):
        """Extract features from a model.

        Args:
            model: PyTorch model (encoder or full model)
            dataloader: DataLoader to extract features from
            device: Device to use (default: 'cuda')
            encoder_type: Type of encoder ('1d', '2d', or 'contrastive')

        Returns:
            Tuple of (features, digit_labels, speaker_ids) as numpy arrays
        """
        model.eval()
        model = model.to(device)

        all_features = []
        all_digits = []
        all_speakers = []

        for batch in tqdm(dataloader, desc=f"Extracting {encoder_type} features"):
            # Move batch to device
            if encoder_type == '1d':
                inputs = batch['waveform'].to(device)
                features = model(inputs)
            elif encoder_type == '2d':
                inputs = batch['spectrogram'].to(device)
                features = model(inputs)
            elif encoder_type == 'contrastive':
                # For contrastive models, we need both inputs
                waveform = batch['waveform'].to(device)
                spectrogram = batch['spectrogram'].to(device)
                # Use get_embeddings if available, otherwise just use encoders
                if hasattr(model, 'get_embeddings'):
                    h1, h2 = model.get_embeddings(waveform, spectrogram)
                    features = (h1, h2)  # Return both for later selection
                else:
                    features = model.encoder_1d(waveform)
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}")

            all_features.append(features.cpu())
            all_digits.append(batch['digit'].cpu())
            all_speakers.append(batch['speaker_id'].cpu())

        # Concatenate all batches
        features = torch.cat(all_features, dim=0).numpy()
        digits = torch.cat(all_digits, dim=0).numpy()
        speakers = torch.cat(all_speakers, dim=0).numpy()

        return features, digits, speakers

    @staticmethod
    @torch.no_grad()
    def extract_contrastive_features(
        contrastive_model,
        dataloader,
        device='cuda',
        feature_type='1d',
    ):
        """Extract features from contrastive model.

        Args:
            contrastive_model: Contrastive model with encoder_1d and encoder_2d
            dataloader: DataLoader to extract features from
            device: Device to use
            feature_type: Which features to extract ('1d', '2d', or 'concat')

        Returns:
            Tuple of (features, digit_labels, speaker_ids) as numpy arrays
        """
        contrastive_model.eval()
        contrastive_model = contrastive_model.to(device)

        all_features = []
        all_digits = []
        all_speakers = []

        for batch in tqdm(dataloader, desc=f"Extracting {feature_type} features"):
            waveform = batch['waveform'].to(device)
            spectrogram = batch['spectrogram'].to(device)

            # Get embeddings
            h1, h2 = contrastive_model.get_embeddings(waveform, spectrogram)

            # Select features based on type
            if feature_type == '1d':
                features = h1
            elif feature_type == '2d':
                features = h2
            elif feature_type == 'concat':
                features = torch.cat([h1, h2], dim=1)
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")

            all_features.append(features.cpu())
            all_digits.append(batch['digit'].cpu())
            all_speakers.append(batch['speaker_id'].cpu())

        # Concatenate all batches
        features = torch.cat(all_features, dim=0).numpy()
        digits = torch.cat(all_digits, dim=0).numpy()
        speakers = torch.cat(all_speakers, dim=0).numpy()

        return features, digits, speakers

    @staticmethod
    def train_and_evaluate(
        X_train,
        y_train,
        X_val,
        y_val,
        max_iter=1000,
        verbose=True,
    ):
        """Train logistic regression and evaluate.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            max_iter: Maximum iterations for solver
            verbose: Whether to print results

        Returns:
            Dictionary with train and validation accuracies
        """
        # Train logistic regression
        clf = LogisticRegression(
            max_iter=max_iter,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42,
            verbose=0,
        )

        clf.fit(X_train, y_train)

        # Evaluate
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        if verbose:
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")

        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'model': clf,
        }

    @staticmethod
    def evaluate_multimodal(
        contrastive_model,
        train_dataloader,
        val_dataloader,
        device='cuda',
    ):
        """Evaluate contrastive model with linear probes on multiple feature types.

        Args:
            contrastive_model: Trained contrastive model
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            device: Device to use

        Returns:
            Dictionary with results for each feature type
        """
        results = {}

        for feature_type in ['1d', '2d', 'concat']:
            print(f"\n{'='*50}")
            print(f"Evaluating linear probe on {feature_type} features")
            print(f"{'='*50}")

            # Extract features
            X_train, y_train, _ = LinearProbe.extract_contrastive_features(
                contrastive_model, train_dataloader, device, feature_type
            )
            X_val, y_val, _ = LinearProbe.extract_contrastive_features(
                contrastive_model, val_dataloader, device, feature_type
            )

            # Train and evaluate
            result = LinearProbe.train_and_evaluate(X_train, y_train, X_val, y_val)
            results[feature_type] = result

        # Print summary
        print(f"\n{'='*50}")
        print("Summary of Linear Probe Results")
        print(f"{'='*50}")
        for feature_type, result in results.items():
            print(f"{feature_type:8s} - Val Acc: {result['val_acc']:.4f}")

        return results

    @staticmethod
    def evaluate_supervised(
        encoder,
        train_dataloader,
        val_dataloader,
        device='cuda',
        encoder_type='1d',
    ):
        """Evaluate supervised encoder with linear probe.

        Args:
            encoder: Trained encoder (just the encoder, not the full model)
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            device: Device to use
            encoder_type: Type of encoder ('1d' or '2d')

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*50}")
        print(f"Evaluating supervised {encoder_type} encoder")
        print(f"{'='*50}")

        # Extract features
        X_train, y_train, _ = LinearProbe.extract_features(
            encoder, train_dataloader, device, encoder_type
        )
        X_val, y_val, _ = LinearProbe.extract_features(
            encoder, val_dataloader, device, encoder_type
        )

        # Train and evaluate
        result = LinearProbe.train_and_evaluate(X_train, y_train, X_val, y_val)

        return result
