from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import joblib
from sklearn import svm
from sklearn import metrics

def eval_SVM():
    path = str(Path(__file__).resolve().parent) + "/"

    # Inputs
    inputs = np.loadtxt(path + "../training_data/NN_input_test.csv", delimiter=',')

    # Targets
    targets = np.loadtxt(path + "../training_data/NN_target_test.csv", delimiter=',')

    # SVM
    model = joblib.load(path + "3x3HeuristicModel.svm")

    # Predict
    pred = np.floor(model.predict(inputs)*3)

    print(f"Accuracy: {metrics.accuracy_score(targets, pred)}")
    print(f"MSE: {metrics.mean_absolute_error(targets, pred)}")

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
    #eval_NN()
    eval_SVM()
