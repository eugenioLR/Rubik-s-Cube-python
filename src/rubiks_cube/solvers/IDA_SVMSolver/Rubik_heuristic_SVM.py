import numpy as np
import numpy.random
import torch
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import joblib
import pickle
import random
from pathlib import Path
import time
from matplotlib import pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

def main():
    path = str(Path(__file__).resolve().parent) + "/"

    inputs = np.loadtxt(path + "../training_data/NN_input_svm.csv", delimiter=',')[1:]
    targets = np.loadtxt(path + "../training_data/NN_target_svm.csv", delimiter=',')[1:]

    targets = targets//3

    # DEBUG ONLY
    #order = list(range(len(inputs)))
    #np.random.shuffle(order)
    #order = order[:225000]
    #order = order[:1000]

    inputs = inputs.astype(int)
    targets = targets.astype(int)

    print(inputs.shape)
    print(targets.shape)


    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3)
    print("Training")
    start = time.time()
    # [2, 1, 0.004448383571631944, 15.974062431147853, 0.6874183153208718, 1.5386042095357761, 0.043163610025160636]
    regr = svm.SVR(kernel='rbf', degree=1, gamma=0.004448, coef0=15.97406, tol=0.68741, C=1.53860, epsilon=0.04316, cache_size=25000, shrinking=False)
    # Mean Squared Error: 13.879358126800081

    # [2, 5, 0.008258233493522544, 28.72336292366309, 0.4084284533561991, 1.7298150797590937, 0.03914126508362224]
    #regr = svm.SVR(kernel='rbf', degree=5, gamma=0.00825, coef0=28.72336, tol=0.40842, C=1.72981, epsilon=0.03914, cache_size=25000, shrinking=False)
    # Mean Squared Error: 13.634005886255121

    # [1, 2, 0.055516999776763415, 18.768706929479, 0.3520188193898568, 0.00741089952774208, 0.21040588502463253]
    #regr = svm.SVR(kernel='poly', degree=2, gamma=0.055516, coef0=18.76870, tol=0.352018, C=0.00741, epsilon=0.21040, cache_size=25000, shrinking=False)
    # Mean Squared Error: 14.36574822474067

    #[1, 8, 2.6379136411992916, 29.200081799143454, 3.919992664997631, 6.801346589983689, 0.7504120899236773]
    #regr = svm.SVR(kernel='poly', degree=8, gamma=2.63791, coef0=29.20008, tol=3.91999, C=6.80135, epsilon=0.75041, cache_size=25000, shrinking=False)
    # Mean Squared Error: 15.202642368042588


    regr.fit(x_train, y_train)
    end = time.time()
    time_spent = end-start

    y_pred = np.floor(regr.predict(x_test))

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average='macro'))
    r2 = metrics.r2_score(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)

    print(f"Training time {time_spent}")
    print("R^2:", r2)
    print("Mean Squared Error:", mse)
    joblib.dump(regr, "3x3HeuristicModel.svm")

if __name__ == '__main__':
    main()
