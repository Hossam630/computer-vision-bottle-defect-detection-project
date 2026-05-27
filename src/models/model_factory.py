# src/models/model_factory.py

import torch.nn as nn
from torchvision import models

# Import custom models
from .dinov2 import DINOv2
from .siglip2 import SigLIP2

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False

def build_model(model_name, num_classes):
    model_name = model_name.lower()

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        freeze(model)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        freeze(model)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        freeze(model)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    elif model_name == "dinov2":
        model = DINOv2(num_classes)

    elif model_name == "siglip2":
        model = SigLIP2(num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
