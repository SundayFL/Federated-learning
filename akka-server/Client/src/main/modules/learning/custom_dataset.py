from genericpath import isfile
import numpy as np
from numpy.lib.npyio import load
from scipy import io as sio
import torch
import os
import pandas as pd

class CustomDataset():

    def __init__(self,  transform=None, imported_data =None):

        self.data = imported_data.data
        self.targets = imported_data.targets.astype('long')

        for i in range(len(self.targets)):
            self.data[i] = transform(self.data[i])
        self.transform = transform
        self.data, self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        target = self.targets[idx]

        return data, target