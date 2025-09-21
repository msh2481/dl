import torch
import torch.nn as nn
from beartype import beartype as typed
from clip import clip
from jaxtyping import Float
from torch import Tensor as TT
from tqdm import tqdm

IMAGENET_TEMPLATES = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]


@typed
def zeroshot_classifier(
    model: torch.nn.Module, classnames: list[str]
) -> Float[TT, "emb_dim n_classes"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Zero-shot classifier"):
            texts = [
                template.format(classname) for template in IMAGENET_TEMPLATES
            ]  # format with class
            texts = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


class CLIPZeroShotClassifier(nn.Module):
    """Zero-shot image classifier built from a CLIP visual backbone."""

    @typed
    def __init__(
        self,
        classnames: list[str],
        backbone: str = "ViT-B/16",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.model, self.preprocess = clip.load(backbone)
        self.dtype = self.model.dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.classnames = list(classnames)
        self.temperature = temperature

        with torch.no_grad():
            zeroshot_weights = zeroshot_classifier(self.model, classnames).to(
                device=self.device, dtype=self.dtype
            )
        self.head = nn.Parameter(zeroshot_weights, requires_grad=True)

    @typed
    def forward(
        self, images: Float[TT, "batch_size 3 h w"]
    ) -> Float[TT, "batch_size n_classes"]:
        images = images.to(device=self.device, dtype=self.dtype)
        features = self.model.visual(images)
        features = features / features.norm(dim=-1, keepdim=True)
        batch_size, _, h, w = images.shape
        assert features.shape[0] == batch_size and features.ndim == 2
        assert self.head.ndim == 2 and self.head.shape[0] == features.shape[1]
        return self.temperature * (features @ self.head)

    @typed
    def get_energy(
        self, images: Float[TT, "batch_size 3 h w"], texts: list[str]
    ) -> Float[TT, "batch_size"]:
        images = images.to(device=self.device, dtype=self.dtype)
        features = self.model.visual(images)
        features = features / features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            texts = clip.tokenize(texts).to(device=self.device)
            text_embeddings = self.model.encode_text(texts)
            text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            )

        assert features.shape == text_embeddings.shape
        return (features * text_embeddings).sum(dim=-1)


if __name__ == "__main__":
    BACKBONE = "ViT-B/16"
    model, preprocess = clip.load(BACKBONE)
    sample_classes = ["dog", "cat", "bird"]
    zeroshot_weights = zeroshot_classifier(model, sample_classes)
    print(zeroshot_weights.shape)
    print(zeroshot_weights)
