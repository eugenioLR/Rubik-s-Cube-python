import time
from pathlib import Path
import random
from Cube import Cube
from rubikNotation import *

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
