import numpy as np
import torch
import random
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class Sign(nn.Module):
    def __init__(self):
        super(Sign, self).__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size = 3,stride = 1,padding = 1) #input: (m,28,28,1) output: (m,28,28,6)
        self.max1 = nn.MaxPool2d(kernel_size = (2,2),stride = 2) #input: (m,28,28,6) output: (m,14,14,6)
        self.conv2 = nn.Conv2d(6,16,kernel_size = 5,stride = 1,padding = 0) #input: (m,14,14,6) output: (m,10,10,16)
        self.max2 = nn.MaxPool2d(kernel_size = (2,2),stride = 2) #input: (m,10,10,16) output: (m,5,5,16)
        self.fc1 = nn.Linear(400,120) #input: (m,400) output: (m,120)
        self.fc2 = nn.Linear(120,84) #input: (m,120) output: (m,84)
        self.fc3 = nn.Linear(84,25) #input: (m,84) output: (m,25)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        x = torch.flatten(x,start_dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)