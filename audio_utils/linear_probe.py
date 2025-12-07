import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class LinearProbe:
    @staticmethod
    @torch.no_grad()
    def extract_features(
        model,
        dataloader,
        device="cuda",
        encoder_type="1d",
    ):
        model.eval()
        model = model.to(device)

        all_features = []
        all_digits = []
        all_speakers = []

        for batch in tqdm(
            dataloader, desc=f"Extracting {encoder_type} features"
        ):
            if encoder_type == "1d":
                inputs = batch["waveform"].to(device)
                features = model(inputs)
            elif encoder_type == "2d":
                inputs = batch["spectrogram"].to(device)
                features = model(inputs)
            elif encoder_type == "contrastive":
                waveform = batch["waveform"].to(device)
                spectrogram = batch["spectrogram"].to(device)
                if hasattr(model, "get_embeddings"):
                    h1, h2 = model.get_embeddings(waveform, spectrogram)
                    features = (h1, h2)
                else:
                    features = model.encoder_1d(waveform)
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}")

            all_features.append(features.cpu())
            all_digits.append(batch["digit"].cpu())
            all_speakers.append(batch["speaker_id"].cpu())

        features = torch.cat(all_features, dim=0).numpy()
        digits = torch.cat(all_digits, dim=0).numpy()
        speakers = torch.cat(all_speakers, dim=0).numpy()

        return features, digits, speakers

    @staticmethod
    @torch.no_grad()
    def extract_contrastive_features(
        contrastive_model,
        dataloader,
        device="cuda",
        feature_type="1d",
    ):
        contrastive_model.eval()
        contrastive_model = contrastive_model.to(device)

        all_features = []
        all_digits = []
        all_speakers = []

        for batch in tqdm(
            dataloader, desc=f"Extracting {feature_type} features"
        ):
            waveform = batch["waveform"].to(device)
            spectrogram = batch["spectrogram"].to(device)

            h1, h2 = contrastive_model.get_embeddings(waveform, spectrogram)

            if feature_type == "1d":
                features = h1
            elif feature_type == "2d":
                features = h2
            elif feature_type == "concat":
                features = torch.cat([h1, h2], dim=1)
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")

            all_features.append(features.cpu())
            all_digits.append(batch["digit"].cpu())
            all_speakers.append(batch["speaker_id"].cpu())

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
        clf = LogisticRegression(
            max_iter=max_iter,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
            verbose=0,
        )

        clf.fit(X_train, y_train)

        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        if verbose:
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")

        return {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "model": clf,
        }

    @staticmethod
    def evaluate_multimodal(
        contrastive_model,
        train_dataloader,
        val_dataloader,
        device="cuda",
    ):
        results = {}

        for feature_type in ["1d", "2d", "concat"]:
            print(f"\n{'='*50}")
            print(f"Evaluating linear probe on {feature_type} features")
            print(f"{'='*50}")

            X_train, y_train, _ = LinearProbe.extract_contrastive_features(
                contrastive_model, train_dataloader, device, feature_type
            )
            X_val, y_val, _ = LinearProbe.extract_contrastive_features(
                contrastive_model, val_dataloader, device, feature_type
            )

            result = LinearProbe.train_and_evaluate(
                X_train, y_train, X_val, y_val
            )
            results[feature_type] = result

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
        device="cuda",
        encoder_type="1d",
    ):
        print(f"\n{'='*50}")
        print(f"Evaluating supervised {encoder_type} encoder")
        print(f"{'='*50}")

        X_train, y_train, _ = LinearProbe.extract_features(
            encoder, train_dataloader, device, encoder_type
        )
        X_val, y_val, _ = LinearProbe.extract_features(
            encoder, val_dataloader, device, encoder_type
        )

        result = LinearProbe.train_and_evaluate(X_train, y_train, X_val, y_val)

        return result
