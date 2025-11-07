import torch
import torch.nn as nn
from torchvision import models


def load_backbone_from_checkpoint(checkpoint_path):
    """Load backbone (ResNet18 with fc=Identity) from any checkpoint format.

    Returns backbone that outputs 512-dim features after avgpool.
    Supports supervised, SimCLR, and BYOL checkpoints.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']

    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()

    # Detect checkpoint type prefix
    if any(k.startswith('online_backbone.') for k in state_dict.keys()):
        prefix = 'online_backbone.'
    elif any(k.startswith('backbone.') for k in state_dict.keys()):
        prefix = 'backbone.'
    else:
        prefix = 'model.'

    # Extract backbone weights
    backbone_state_dict = {
        k.replace(prefix, ''): v
        for k, v in state_dict.items()
        if k.startswith(prefix) and not k.replace(prefix, '').startswith('fc.')
    }

    backbone.load_state_dict(backbone_state_dict)
    return backbone


def load_model_for_classification(checkpoint_path, num_classes=10):
    """Load full classification model from any checkpoint format.

    For supervised: loads full model with fc layer
    For SimCLR/BYOL: loads backbone + new fc layer (randomly initialized)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']

    # Detect checkpoint type
    if any(k.startswith('online_backbone.') for k in state_dict.keys()) or \
       any(k.startswith('backbone.') for k in state_dict.keys()):
        # SimCLR or BYOL: load backbone + fc
        model = models.resnet18(weights=None, num_classes=num_classes)
        backbone = load_backbone_from_checkpoint(checkpoint_path)
        backbone_state = backbone.state_dict()
        model.load_state_dict(backbone_state, strict=False)

        # Load fc layer if exists in checkpoint
        if 'fc.weight' in state_dict and 'fc.bias' in state_dict:
            model.fc.weight.data = state_dict['fc.weight']
            model.fc.bias.data = state_dict['fc.bias']
    else:
        # Supervised: load full model with correct num_classes
        model = models.resnet18(weights=None, num_classes=num_classes)
        model_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        model.load_state_dict(model_state_dict)

    return model
