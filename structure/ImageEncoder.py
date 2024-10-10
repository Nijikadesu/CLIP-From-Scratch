import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Image Encoder, using ResNet as Backbone.
class ResidualBlock(nn.Module):
    """
    A simple implementation of a Residual Block.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        residual_x = x
        x = F.relu(self.conv3(x) + residual_x)
        # print(x.shape)
        return x


class ImageEncoderResNet(nn.Module):
    """
    Image Encoder using ResNet as backbone.
    """
    def __init__(self):
        super().__init__()
        self.res_block1 = ResidualBlock(in_channels=1, out_channels=16, stride=2)
        self.res_block2 = ResidualBlock(in_channels=16, out_channels=4, stride=2)
        self.res_block3 = ResidualBlock(in_channels=4, out_channels=1, stride=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(49, 8)
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.norm(self.linear(x))

        return x

if __name__ == '__main__':
    model = ImageEncoderResNet()
    print(model)
    input = torch.randn(3, 1, 28, 28)
    output = model(input)
    print(output.shape)