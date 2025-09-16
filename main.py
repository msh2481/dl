from clip import clip
from datasets import load_dataset

BACKBONE = "ViT-B/16"
model = clip.load(BACKBONE)

dataset = load_dataset("wltjr1007/DomainNet")
