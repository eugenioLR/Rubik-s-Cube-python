from pathlib import Path
import torch
import numpy as np
from Rubik_heuristic_NN import NeuralNetwork
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import metrics

def check_distance_hist():
    plt.style.use('seaborn')
    path = str(Path(__file__).resolve().parent) + "/"
    targets = np.loadtxt(path + "../training_data/NN_target_full.csv", delimiter=',').astype(np.int32)
    values, counts = np.unique(targets, return_counts=True)
    plt.bar(values, counts)
    plt.xticks(range(21))
    plt.ylabel("cantidad cubos")
    plt.xlabel("distancia")
    plt.title("diagrama de barras de los datos")
    plt.show()


def eval_NN():
    path = str(Path(__file__).resolve().parent) + "/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inputs
    cube_data = np.loadtxt(path + "../training_data/NN_input_test.csv", delimiter=',')
    cube_data = torch.from_numpy(cube_data)

    colors_in_face = []
    for i in range(len(cube_data)):
        colors_in_face.append(torch.tensor([len(torch.unique(i)) for i in torch.reshape(cube_data[i], [6, 9])]))
    colors_in_face = torch.stack(colors_in_face)
    inputs = torch.cat([cube_data, colors_in_face], dim=1).float().to(device)
    #inputs = cube_data.float().to(device)

    # Targets
    targets = np.loadtxt(path + "../training_data/NN_target_test.csv", delimiter=',')
    targets = torch.tensor(targets).to(device)

    # Neural Network
    model = torch.load(path + "3x3HeuristicModel.pt").to(device)
    model.eval()

    # Predict
    pred_one_hot = model(inputs)
    pred = torch.argmax(pred_one_hot, dim=1)
    #pred = 2*pred_one_hot
    targets = targets.cpu().numpy()
    pred = pred.detach().cpu().numpy()

    print(f"Accuracy: {metrics.accuracy_score(targets, pred)}")
    print(f"MAE: {metrics.mean_absolute_error(targets, pred)}")

    # Plot predictions against real values
    plt.style.use('seaborn')
    plt.plot(targets, pred, 'o')
    plt.plot([0, 20], [0, 20])
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.xticks(range(21))
    plt.yticks(range(21))
    plt.show()

if __name__ == '__main__':
    eval_NN()
    #check_distance_hist()
