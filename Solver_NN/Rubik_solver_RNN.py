import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import numpy.matlib

# NOT FULLY IMPLEMENTED

# https://portfolios.cs.earlham.edu/wp-content/uploads/2019/05/Combining_heuristics_with_neural_networks__488_3.pdf

move_to_int = {
    'U':0, "U'":1, 'U2':2,
    'D':3, "D'":4, 'D2':5,
    'R':6, "R'":7, 'R2':8,
    'L':9, "L'":10,'L2':11,
    'F':12,"F'":13,'F2':14,
    'B':13,"B'":16,'B2':17,
}

class RNN(nn.Module):
    """
    Possible moves:
    ['U','D','R','L','F','B'] + ""
    ['U','D','R','L','F','B'] + "'"
    ['U','D','R','L','F','B'] + "2"

    = 6*3 = 18

    Neural network with 3 hidden layers of the following sizes:
    54 -> 40 -> 35 -> 30 -> 18
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        #self.layer_sizes = [
        #    input_size,
        #    int(input_size*0.35),
        #    int(input_size*0.25),
        #    int(input_size*0.15),
        #    output_size
        #]

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        print(input_tensor.shape)
        print(hidden_tensor.shape)

        combined = torch.cat([input_tensor, hidden_tensor], dim=1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape([1, x.shape[0]], -1)
            y = y.reshape([1, y.shape[0]], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            _, real = y.max(1)
            #num_correct += ((predictions - real)/y.shape[1])**2
            num_correct += (predictions == real).int()
            num_samples += 1

        acc = float(num_correct)/float(num_samples)

    model.train()
    return acc

def main():
    #category_lines, all_cateories = load_data()
    #n_categories = len(all_cateories)

    n_hidden = 30
    model = RNN(54, n_hidden, 18)

    input_tensor = torch.zeros(54).reshape([1, 54])
    hidden_tensor = model.init_hidden()

    output, next_hidden = model(input_tensor, hidden_tensor)

if __name__ == "__main__":
    main()
