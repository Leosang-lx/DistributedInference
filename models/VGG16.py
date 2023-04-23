import torch
import torch.nn as nn
from torch import Tensor
from models.googlenet import BasicConv2d

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


# class Conv2dReLU(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
#         super(Conv2dReLU, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
#         # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x: Tensor) -> Tensor:
#         out = self.conv(x)
#         # out = self.bn(out)
#         out = self.relu(out)
#
#         return out


a = [
    BasicConv2d(3, 64, kernel_size=3, padding=1),
    BasicConv2d(64, 64, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    BasicConv2d(64, 128, kernel_size=3, padding=1),
    BasicConv2d(128, 128, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    BasicConv2d(128, 256, kernel_size=3, padding=1),
    BasicConv2d(256, 256, kernel_size=3, padding=1),
    BasicConv2d(256, 256, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    BasicConv2d(256, 512, kernel_size=3, padding=1),
    BasicConv2d(512, 512, kernel_size=3, padding=1),
    BasicConv2d(512, 512, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    BasicConv2d(512, 512, kernel_size=3, padding=1),
    BasicConv2d(512, 512, kernel_size=3, padding=1),
    BasicConv2d(512, 512, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2)
]


class vgg16(nn.Module):
    def __init__(self, num_classes=1000):
        super(vgg16, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(64, 128, kernel_size=3, padding=1),
            BasicConv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(128, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(256, 512, kernel_size=3, padding=1),
            BasicConv2d(512, 512, kernel_size=3, padding=1),
            BasicConv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(512, 512, kernel_size=3, padding=1),
            BasicConv2d(512, 512, kernel_size=3, padding=1),
            BasicConv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layers = [*self.features]
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self.layers = [*self.features]
        self.input_shape = 3, 224, 224
        self.depth = len(self.features)
        self.next = [*list(range(1, self.depth)), []]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward_feature(self, x):
        return self.features(x)

    def forward_classifier(self, x):
        return self.classifier(torch.flatten(x))