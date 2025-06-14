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

class NeuralNetwork(nn.Module):
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

def check_accuracy(loader, model, device='cuda', batch_size = 200):
    num_correct = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        num_samples = len(loader.dataset)
        x_var = np.zeros(num_samples)
        y_var = np.zeros(num_samples)
        for batch_id, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            y = y.reshape(y.shape[0], -1)

            scores = model(x)
            predictions = scores.argmax(1)
            real = y.argmax(1)

            indices = np.arange(batch_id*batch_size, batch_id*batch_size + x.shape[0])
            x_var[indices] = predictions.int().cpu()
            y_var[indices] = real.int().cpu()

            num_correct += float((predictions == real).int().sum())

        acc = float(num_correct)/float(num_samples)

    model.train()
    return acc

def show_stats(loader, model, device='cuda', batch_size = 200):
    num_correct = 0
    num_samples = 0
    value_sum = 0
    square_error_sum = 0

    model.eval()
    with torch.no_grad():
        num_samples = len(loader.dataset)
        x_var = np.zeros(num_samples)
        y_var = np.zeros(num_samples)

        for batch_id, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            y = y.reshape(y.shape[0], -1)

            scores = model(x)
            predictions = scores.argmax(1)
            real = y.argmax(1)

            indices = np.arange(batch_id*batch_size, batch_id*batch_size + x.shape[0])
            x_var[indices] = predictions.int().cpu()
            y_var[indices] = real.int().cpu()

            square_error_sum += ((predictions - real)**2).sum()

            num_correct += float((predictions == real).int().sum())


        pred_range = (x_var.min(), x_var.max())
        pred_mse = torch.sqrt(square_error_sum/num_samples)
        pred_acc = num_correct/num_samples

        print("STATS:")
        print(f"\tRange predictions: [{pred_range[0]} - {pred_range[1]}]")
        print(f"\tMean squared error: {pred_mse}")
        print(f"\tAccuracy: {pred_acc}")

    model.train()
    return pred_range, pred_mse, pred_acc


def setup_dataflow(dataset, train_idx, val_idx, batch=200):
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=batch, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch, sampler=val_sampler)

    return train_loader, val_loader

def main():
    path = str(Path(__file__).resolve().parent) + "/"

    ## Plot config
    plt.style.use('seaborn')

    ## Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Load training data
    # Inputs
    cube_data = np.loadtxt(path + "../training_data/NN_input_nn.csv", delimiter=',')

    #REDUCE SIZE ONLY DEBUG
    np.random.shuffle(cube_data)
    cube_data = cube_data[:20000, :]

    # Amount of colors in each face
    colors_in_face = []
    for i in range(len(cube_data)):
        colors_in_face.append(np.array([len(np.unique(face)) for face in np.reshape(cube_data[i], [6, 9])]))
    colors_in_face = np.stack(colors_in_face)

    inputs = np.hstack([cube_data, colors_in_face])
    inputs = torch.tensor(inputs).float().to(device)

    # Targets
    targets_original = np.loadtxt(path + "../training_data/NN_target_nn.csv", delimiter=',')

    #REDUCE SIZE ONLY DEBUG
    np.random.shuffle(targets_original)
    targets_original = targets_original[:20000]

    targets_original = torch.tensor(targets_original).long()
    targets = F.one_hot(targets_original).to(device)


    ## Parameters
    input_len = inputs.shape[1]
    target_len = targets.shape[1]
    learning_rate = 0.001
    batch_size = 1024
    num_epochs = 1000

    # Data loader
    print(f"Input tensor size: {inputs.shape}")
    print(f"Target tensor size: {targets.shape}")
    dataset = TensorDataset(inputs, targets)

    num_splits = 10
    splits = KFold(n_splits=num_splits, shuffle=True)

    split_amount = [int(len(dataset)*0.7), int(len(dataset)*0.15), int(len(dataset)*0.15)]
    split_amount[0] += len(dataset) - sum(split_amount)


    train_data, test_data, val_data = torch.utils.data.random_split(dataset, split_amount)
    #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ## Initialize network
    model = NeuralNetwork(input_len, target_len).to(device)
    print(f"NN architecture: {model.layer_sizes}")

    ## Loss and optimizer
    # weights
    values, counts = np.unique(targets_original, return_counts=True)
    weights = targets_original.shape[0]/(len(values) * counts)
    weights = torch.tensor(weights).float().to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(weight=weights)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler, for decreasing the learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    train_history = []
    test_history = []
    val_history = []
    mse_history = []

    info_flag = True
    val_acc = 0
    aux_val = 0
    counter = 0
    max_count = 25
    stall_tol = 0.0001

    if info_flag:
        fig, ax1, ax2 = setup_dynamic_plot()

    try:
        ## Training
        for i in range(num_epochs):
            aux_val = val_acc
            val_acc = check_accuracy(val_loader, model, device)
            if abs(aux_val - val_acc) < stall_tol:
                counter += 1
            else:
                counter = 0

            if counter >= max_count:
                break

            test_acc = 0
            test_count = 0
            # With cross validation
            for split_idx, (train_idx, test_idx) in enumerate(splits.split(np.arange(len(dataset)))):

                train_loader_split, test_loader_split = setup_dataflow(dataset, train_idx, test_idx, batch_size)

                # Train with each permutation of the slipt
                for batch_id, (dat, tar) in enumerate(train_loader_split):
                    dat = dat.to(device=device, non_blocking=True)
                    tar = tar.to(device=device, non_blocking=True)

                    # Datatypes
                    dat = dat.reshape(dat.shape[0], -1)
                    tar = tar.reshape(tar.shape[0], -1)

                    # Forward
                    score = model(dat)

                    loss = criterion(score, tar.argmax(1))

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()

                    #gradient descent
                    optimizer.step()


                if info_flag:
                    test_acc += check_accuracy(test_loader_split, model, device)

            test_history.append(test_acc)
            print()
            print(f"Epoch: {i}, stall: {counter}")
            if info_flag:
                pred_range, mse, acc = show_stats(test_loader, model, device, batch_size)
                val_history.append(acc)
                mse_history.append(mse)
                dynamic_display_acc(fig, ax1, ax2, test_history, val_history, mse_history)

            scheduler.step()

    except KeyboardInterrupt:
        print("\nInterrumpted training, saving model and showing statistics")

    #if info_flag:
        #display_acc(test_history, mse_hist = dev_history)

    # Check accuracy
    print("On all data:")
    show_stats(dataset_loader, model, device)

    if input("save model?(Y/N): ").upper() == 'Y':
        torch.save(model, path + '3x3HeuristicModel.pt')
        print("Model saved")

def setup_dynamic_plot():
    fig = plt.figure()
    plt.ion()

    ax1 = plt.subplot(1,2,1)
    ax1.plot([0,0])
    ax1.plot([0,0])
    #plt.draw()

    ax2 = plt.subplot(1,2,2)
    ax2.plot([0,0])
    #plt.draw()

    plt.show()

    return fig, ax1, ax2



def dynamic_display_acc(fig, ax1, ax2, test_hist, val_hist, mse_hist):

    #ax1 = plt.subplot(1,2,1)
    #plt.plot(test_hist, "r")
    #plt.plot(val_hist, "g")
    plt.subplot(1,2,1)
    ax1.lines[0].set_data(np.arange(0, len(test_hist)), test_hist)
    ax1.lines[1].set_data(np.arange(0, len(test_hist)),val_hist)
    ax1.relim()
    ax1.autoscale_view()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy")
    ax1.legend(["test data", "all data"])
    fig.canvas.draw_idle()#plt.draw()

    plt.subplot(1,2,2)
    #ax2 = plt.subplot(1,2,2)
    #plt.plot(mse_hist, "b")
    ax2.lines[0].set_data(np.arange(0, len(test_hist)),mse_hist)
    ax2.relim()
    ax2.autoscale_view()
    plt.xlabel("epoch")
    plt.ylabel("sqrt(MSE)")
    plt.title("Square root of Mean Squared Error")
    fig.canvas.draw_idle()#plt.draw()
    fig.canvas.flush_events()

    plt.pause(0.000001)


def display_acc(test_hist, train_hist=None, mse_hist=None):
    plt.plot(test_hist)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    if train_hist is not None:
        plt.plot(train_hist)
        plt.legend(["test data", "train data"])
    plt.show()

    if mse_hist is not None:
        plt.plot(mse_hist)
        plt.legend("squared error")
        plt.xlabel("epoch")
        plt.ylabel("error")
        plt.show()

if __name__ == '__main__':
    main()
