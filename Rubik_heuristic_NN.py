import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from get_NN_train_data import *



data = Rubik_train_data(2, 11, 1000000)
data.prepare_data()
data_purged = data.cleanup_data()
print(f"got rid of {data_purged} data points")
data.linearlize_data()

inputs = data.get_net_inputs()
targets = data.get_net_targets()


with open("NN_input.txt", "w") as file_in:
    for i in inputs:
        file_in.write(" ".join([str(j) for j in i]))
        file_in.write("\n")

with open("NN_target.txt", "w") as file_targ:
    file_targ.write(str(targets[0]))
    for i in targets[1:]:
        file_targ.write(",")
        file_targ.write(str(i))


input_len = len(inputs[0])
#target_len = len(targets[0])


"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_len, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 5),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#exit(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
model = NeuralNetwork().to(device)
print(model)
"""
