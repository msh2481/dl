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
        for classname in tqdm(classnames):
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
        temperature: float = 100.0,
    ) -> None:
        super().__init__()
        model, self.preprocess = clip.load(backbone)
        self.visual = model.visual
        self.transformer = model.transformer

        frozen_model, _ = clip.load(backbone)
        self.original_visual = frozen_model.visual
        for param in self.original_visual.parameters():
            param.requires_grad = False

        self.dtype = model.dtype
        self.classnames = list(classnames)
        self.temperature = temperature

        with torch.no_grad():
            zeroshot_weights = zeroshot_classifier(model, classnames).to(model.dtype)
        self.head = nn.Parameter(zeroshot_weights, requires_grad=False)

    def forward(
        self, images: Float[TT, "batch_size 3 h w"]
    ) -> Float[TT, "batch_size n_classes"]:
        images = images.to(device=self.head.device, dtype=self.dtype)
        features = self.visual(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return self.temperature * (features @ self.head)


if __name__ == "__main__":
    BACKBONE = "ViT-B/16"
    model, preprocess = clip.load(BACKBONE)
    sample_classes = ["dog", "cat", "bird"]
    zeroshot_weights = zeroshot_classifier(model, sample_classes)
    print(zeroshot_weights.shape)
    print(zeroshot_weights)
