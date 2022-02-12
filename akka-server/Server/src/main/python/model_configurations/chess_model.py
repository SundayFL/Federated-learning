import numpy as np
import torch
import random
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class Chess(nn.Module):
    def __init__(self):
        super(Chess, self).__init__()
        self.fc1 = nn.Linear(64*64, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 5)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 64*64)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigm(x)