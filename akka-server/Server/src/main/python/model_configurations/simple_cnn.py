import numpy as np
import torch
import random
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Model
class CNN(nn.Module):
    def __init__(self, numChannels, classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20,
            kernel_size= (3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=5,
            kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=5 * 53 * 53, out_features=500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output#F.log_softmax(x, dim=1)