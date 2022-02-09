



import numpy as np
import torch
import random
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

#one secret layer (256) with relu activation and sigmoid activation for BCE Loss
class MIMIC(nn.Module):
    def __init__(self):
        super(MIMIC, self).__init__()
        self.fc1 = nn.Linear(48*19, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):
        x = x.view(-1, 48*19)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


