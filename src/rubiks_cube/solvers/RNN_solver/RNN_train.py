import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, Subset

#import sklearn
from sklearn.model_selection import KFold

import numpy as np
import numpy.matlib

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pathlib import Path

# The maximum accuracy reached was 0.17
# With cross validation: 0.23

class RecNeuralNetwork(nn.Module):
    """
    Neural network with 3 hidden layers of the following sizes:
    54 -> 40 -> 35 -> 30 -> 20
    """
    def __init__(self, input_size, num_classes = 20):
        super(NeuralNetwork, self).__init__()

        self.layer_sizes = [
            input_size,
            40,
            35,
            30,
            num_classes
        ]

        self.softmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(input_size,          self.layer_sizes[1])
        self.fc2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.fc3 = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        self.fc4 = nn.Linear(self.layer_sizes[3], num_classes)
    
    def forward(self, x):
        # layer 1
        x = F.logsigmoid(self.fc1(x))
        #x = F.relu(self.fc1(x))

        # layer 2
        x = F.logsigmoid(self.fc2(x))
        #x = F.relu(self.fc2(x))

        # layer 3
        x = F.logsigmoid(self.fc3(x))
        #x = F.relu(self.fc3(x))

        # output
        x = self.fc4(x)
        return x