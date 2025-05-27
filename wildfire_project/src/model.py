# from torchvision import models
# import torch.nn as nn
# from .config import NUM_CLASSES, DEVICE
#
# def get_model():
#     print("[MODEL] Loading VGG16 model...")
#     model = models.vgg16(pretrained=True)
#     model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
#     print("[MODEL] Final layer adjusted for", NUM_CLASSES, "classes.")
#     return model.to(DEVICE)

from torchvision import models
from torchvision.models import VGG16_Weights
import torch.nn as nn
from .config import NUM_CLASSES, DEVICE

def get_model():
    print("[MODEL] Loading VGG16 model with pretrained weights...")
    weights = VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    print("[MODEL] Final layer adjusted for", NUM_CLASSES, "classes.")
    return model.to(DEVICE)
