import torch
import torch.nn as nn
class ResCNN(nn.Module):
    def __init__(self, data_num=512):
        super(ResCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, 5, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            BasicBlockall(),
            nn.Conv1d(32 * 3, 32, 1, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(32 * data_num, data_num)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        t = self.model(x).view(x.shape[0], -1)
        return self.linear(t)

class Res_BasicBlock(nn.Module):
    def __init__(self, kernelsize, stride=1):
        super(Res_BasicBlock, self).__init__()
        self.bblock = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=kernelsize, stride=stride, padding="same"), nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, kernel_size=kernelsize, stride=1, padding="same"), nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=kernelsize, stride=1, padding="same"), nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        # Through the convolutional layer``
        out = self.bblock(inputs)
        identity = inputs

        output = torch.add(out, identity)  # layers下面有一个add，把这2个层添加进来相加。

        return output


class BasicBlockall(nn.Module):
    def __init__(self, stride=1):
        super(BasicBlockall, self).__init__()

        self.bblock3 = nn.Sequential(Res_BasicBlock(3),
                                     Res_BasicBlock(3)
                                     )

        self.bblock5 = nn.Sequential(Res_BasicBlock(5),
                                     Res_BasicBlock(5)
                                     )

        self.bblock7 = nn.Sequential(Res_BasicBlock(7),
                                     Res_BasicBlock(7)
                                     )

    def forward(self, inputs):
        out3 = self.bblock3(inputs)
        out5 = self.bblock5(inputs)
        out7 = self.bblock7(inputs)

        out = torch.cat((out3, out5, out7), axis=1)
        return out