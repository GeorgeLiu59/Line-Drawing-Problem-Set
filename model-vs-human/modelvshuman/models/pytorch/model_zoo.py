#!/usr/bin/env python3
import torch
import os

from ..registry import register_model
from ..wrappers.pytorch import PytorchModel
import torch.nn as nn
from torchvision import models


def model_pytorch(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__[model_name](pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)

@register_model("pytorch")
def resnet18_base(model_name="resnet18_base", *args):
    """ResNet-18 model trained on photos."""
    model = models.resnet18(pretrained=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom', 'resnet18_base.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return PytorchModel(model, model_name, *args)

@register_model("pytorch")
def resnet18_linecolor(model_name="resnet18_linecolor", *args):
    """ResNet-18 model trained on line drawings and then trained on photos."""
    model = models.resnet18(pretrained=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom', 'resnet18_linecolor.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return PytorchModel(model, model_name, *args)
