import torch.nn as nn
import torchvision.models as models


class ClassificationModel(nn.Module):
    """Base model with convolutional layers (Resnet34)
    """

    def __init__(self, out_dim, dataset: str = ''):
        super(ClassificationModel, self).__init__()
        self.resnet_model = models.resnet34(pretrained=True, num_classes=out_dim)

        self.backbone = self.resnet_model

    def forward(self, x):
        return self.backbone(x)