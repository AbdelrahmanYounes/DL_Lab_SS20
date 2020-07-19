import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=5):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.conv1 = nn.Conv2d(in_channels=history_length, kernel_size=7, out_channels=32, padding=0, stride=4)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, kernel_size=5, out_channels=64, padding=2, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, kernel_size=3, out_channels=128, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, kernel_size=3, out_channels=256, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=256, kernel_size=3, out_channels=128, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=128, kernel_size=3, out_channels=64, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 11 * 11, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, x):
        # TODO: compute forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        #x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.bn3(x)
        # x = self.pool3(x)
        # Remember to flatten the feature map using:
        # x = x.view(batch_size, dim)
        #print(x.size())
        x = x.view(-1, 64 * 11 * 11)
        x = F.dropout(x, p=0.5)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

