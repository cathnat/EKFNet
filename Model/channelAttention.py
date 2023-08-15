import torch.nn as nn
class ChannelAttention(nn.Module):
    def __init__(self,pool, in_planes=1, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(pool)
        self.max_pool = nn.AdaptiveMaxPool2d(pool)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        # self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        x = x.reshape(1, 1, 1, -1)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # out = self.sigmoid(out)
        out = self.flatten(out)
        return out