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

# The maximum accuracy reached was 0.17

class NeuralNetwork(nn.Module):
    """
    Neural network with 3 hidden layers of the following sizes:
    54 -> 40 -> 35 -> 30 -> 24
    """
    def __init__(self, input_size, num_classes = 20):
        super(NeuralNetwork, self).__init__()

        self.layer_sizes = [
            input_size,
            int(input_size*0.35),
            int(input_size*0.25),
            int(input_size*0.15),
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
            predictions = scores.softmax(dim=1).argmax(1)
            real = y.argmax(1)
            num_correct += (predictions == real).int()
            num_samples += 1

        acc = float(num_correct)/float(num_samples)

    model.train()
    return acc

def check_correlation(loader, model, device='cuda'):
    num_correct = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        x_var = np.zeros(len(loader))
        y_var = np.zeros(len(loader))
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape([1, x.shape[0]], -1)
            y = y.reshape([1, y.shape[0]], -1)

            scores = model(x)
            predictions = scores.argmax(1)
            real = y.argmax(1)

            x_var[num_samples] = float(predictions)
            y_var[num_samples] = float(real)

            num_samples += 1

        acc = (x_var * y_var).mean() - x_var.mean()*y_var.mean()
        acc = acc/np.sqrt(x_var.var()*y_var.var())

    model.train()
    return acc

def main():
    ## Plot config
    plt.style.use('seaborn')

    ## Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Load training data
    # Inputs
    cube_data = np.loadtxt("NN_input.csv", delimiter=',')
    cube_data = torch.from_numpy(cube_data)

    # Amount of colors in each face
    colors_in_face = []
    for i in range(len(cube_data)):
        colors_in_face.append(torch.tensor([len(torch.unique(i)) for i in torch.reshape(cube_data[i], [6, 9])]))
    colors_in_face = torch.stack(colors_in_face)

    # Corners
    indices = torch.tensor([0,2,6,8])
    indices = torch.cat([indices+(9*i) for i in range(6)])
    corners = cube_data[:,indices]

    inputs = torch.cat([cube_data, colors_in_face], dim=1).float()
    #inputs = torch.cat([corners, colors_in_face], dim=1).float()
    #inputs = colors_in_face.float()

    # Targets
    targets1 = np.loadtxt("NN_target.csv", delimiter=',')
    range_rep = np.transpose(np.matlib.repmat(np.arange(1,21), len(targets1), 1))
    targets = np.equal(np.matlib.repmat(targets1, 20, 1), range_rep)
    targets = np.transpose(targets)
    targets = torch.from_numpy(targets).long()

    ## Parameters
    input_len = inputs.shape[1]
    target_len = targets.shape[1]
    learning_rate = 0.001
    batch_size = 200
    num_epochs = 1000

    # Data loader
    print(inputs.shape)
    print(targets.shape)
    dataset = TensorDataset(inputs, targets)

    split_amount = [int(len(dataset)*0.7), int(len(dataset)*0.15), int(len(dataset)*0.15)]
    split_amount[0] += len(dataset) - sum(split_amount)

    train_data, test_data, val_data = torch.utils.data.random_split(dataset, split_amount)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    train_check = torch.utils.data.Subset(train_data, torch.arange(int(len(dataset)*0.15)))

    ## Initialize network
    model = NeuralNetwork(input_len, target_len).to(device=device)
    print(model.layer_sizes)

    ## Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

    train_history = []
    test_history = []
    dev_history = []

    info_flag = True
    val_acc = 0
    aux_val = 0
    counter = 0
    max_count = 25
    stall_tol = 0.0001
    best_corr = -100
    best_params = None

    try:
        ## Training
        for i in range(num_epochs):
            aux_val = val_acc
            val_acc = check_accuracy(val_data, model, device)
            if abs(aux_val - val_acc) < stall_tol:
                counter += 1
            else:
                counter = 0

            if counter >= max_count:
                break

            for batch_id, (dat, tar) in enumerate(train_loader):
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

            test_acc = check_accuracy(test_data, model, device)
            test_history.append(test_acc)
            if info_flag:
                train_acc = check_accuracy(train_check, model, device)
                test_dev = check_correlation(test_data, model, device)

                train_history.append(train_acc)
                dev_history.append(test_dev)

                print("epoch:", i, " corr:", test_dev, " stall:", counter, " acc:", test_acc)

                if best_corr < test_dev:
                    best_params = model.state_dict()
                    best_corr = test_dev


            scheduler.step()

    except KeyboardInterrupt:
        print("\nInterrumpted training, saving model and showing statistics")

    if info_flag:
        display_acc(test_history, train_history, dev_history)

    if input("save model?(Y/N): ").upper() == 'Y':
        best_model = NeuralNetwork(input_len, target_len).to(device)
        best_model.load_state_dict(best_params)
        best_model.eval()

        # Check accuracy
        print("On training data:")
        print("accuracy", check_accuracy(train_data, best_model, device))
        print("correalation:", check_correlation(train_data, best_model, device))

        print("\nOn test data:")
        print("accuracy:", check_accuracy(test_data, best_model, device))
        print("correalation:", check_correlation(test_data, best_model, device))

        print("\nOn validation data:")
        print("accuracy:", check_accuracy(val_data, best_model, device))
        print("correalation:", check_correlation(val_data, best_model, device))

        torch.save(best_model, '3x3HeuristicModel.pt')
        print("Model saved")

def display_acc(test_hist, train_hist=[0, 0], dev_hist=[0,0]):
    plt.plot(test_hist)
    plt.plot(train_hist)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["test data", "train data"])
    plt.show()

    plt.plot(dev_hist)
    plt.legend("squared error")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.show()

if __name__ == '__main__':
    main()
