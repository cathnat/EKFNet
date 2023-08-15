import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseSeparableConv_fw(nn.Module):
    def __init__(self,input_num):
        super(DepthwiseSeparableConv_fw, self).__init__()
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(1, 1, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(1, 1, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(9,4)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        x = x.reshape(1,1,4)
        return x