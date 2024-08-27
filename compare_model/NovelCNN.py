import torch.nn as nn
import torch.nn.functional as F

class NovelCNN(nn.Module):
    def __init__(self, input_size=(1024, 1)):
        super(NovelCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AvgPool1d(kernel_size=2)

        self.conv7 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.AvgPool1d(kernel_size=2)

        self.conv9 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.drop5 = nn.Dropout(0.5)
        self.pool5 = nn.AvgPool1d(kernel_size=2)

        self.conv11 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.drop6 = nn.Dropout(0.5)
        self.pool6 = nn.AvgPool1d(kernel_size=2)

        self.conv13 = nn.Conv1d(1024, 2048, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv1d(2048, 2048, kernel_size=3, stride=1, padding=1)
        self.drop7 = nn.Dropout(0.5)
        self.pool7 = nn.AvgPool1d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8192, 512)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.drop4(x)
        x = self.pool4(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.drop5(x)
        x = self.pool5(x)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.drop6(x)
        x = self.pool6(x)

        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = self.drop7(x)
        x = self.pool7(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x