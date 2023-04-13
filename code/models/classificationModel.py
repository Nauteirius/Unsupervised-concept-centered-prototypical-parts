import torch.nn as nn
import torchvision.models as models


class ClassificationModel(nn.Module):
    """Base model with convolutional layers (Resnet34)
    """

    def __init__(self, out_dim, dataset: str = ''):
        super(ClassificationModel, self).__init__()
        self.resnet_model = models.resnet34(pretrained=True, num_classes=out_dim)

        self.backbone = self.resnet_model
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.backbone(x)