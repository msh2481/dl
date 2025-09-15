import torch
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
def zeroshot_classifier(model: torch.nn.Module, classnames: list[str]) -> Float[TT, "emb_dim n_classes"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in IMAGENET_TEMPLATES] #format with class
            texts = clip.tokenize(texts).to(device) 
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

if __name__ == "__main__":
    BACKBONE = "ViT-B/16"
    model, preprocess = clip.load(BACKBONE)
    sample_classes = ["dog", "cat", "bird"]
    zeroshot_weights = zeroshot_classifier(model, sample_classes)
    print(zeroshot_weights.shape)
    print(zeroshot_weights)