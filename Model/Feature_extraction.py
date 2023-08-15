import torch
import torch.nn as nn
from Model.DepthwiseSeparableConv import DepthwiseSeparableConv

class ChannelAttention(nn.Module):
    def __init__(self,pool, in_planes=4, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(pool)
        self.max_pool = nn.AdaptiveMaxPool2d(pool)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return out.view(1,4,1,16)



class SPPCSPC(nn.Module):
    def __init__(self, c1=1, k=5):
        super(SPPCSPC, self).__init__()
        c_ = 4
        self.cv1 = DepthwiseSeparableConv(c1, c_, 3, padding=1, stride=2)
        self.cv2 = Conv(c1, c_, 1, 2)
        self.ca = ChannelAttention(4)
        self.m1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, (3, 1), 1)
        self.cv7 = Conv(c_, c_, (1, 3), 1)
        self.cv8 = Conv(2 * c_, 1, 1, 1)
        self.fc=nn.Linear(16, 4)

    def forward(self, x):
        x = x.reshape(1, 1, 1, -1)
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x2 = self.ca(x2)
        MaxPool1 = self.m1(x1)
        MaxPool2 = self.m2(MaxPool1)
        MaxPool3 = self.m3(MaxPool2)
        y1 = torch.cat((x1, MaxPool1, MaxPool2, MaxPool3), dim=1)
        y1 = self.cv7(self.cv6(self.cv5(y1)))
        out=self.cv8(torch.cat((y1, x2), dim=1))
        out=self.fc(out)
        out = out.reshape([1,1,4])
        return out

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

