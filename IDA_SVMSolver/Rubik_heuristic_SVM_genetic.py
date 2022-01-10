import numpy as np
import numpy.random
import torch
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from joblib import dump, load
import pickle
import random
from pathlib import Path
import time
from matplotlib import pyplot as plt
from warnings import simplefilter

from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def init_data(test_size = 0.3):
    path = str(Path(__file__).resolve().parent) + "/"

    cube_data = np.loadtxt(path + "../training_data/NN_input_GA.csv", delimiter=',')
    targets = np.loadtxt(path + "../training_data/NN_target_GA.csv", delimiter=',')//3

    inputs = cube_data.astype(np.int32)
    targets = targets.astype(np.int32)

    print(inputs.shape)
    print(targets.shape)

    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_size)

    return x_train, x_test, y_train, y_test

kernel_idx = {0:'linear', 1:'poly', 2:'rbf', 3:'sigmoid'}

def general_logistic(x, min_value = 0, max_val = 1, midpoint = 0, growth = 1):
    val_range = max_val - min_value
    return min_value + val_range / (1 + np.exp(-growth * (x - midpoint)))

def test_svm(params, x_train, x_test, y_train, y_test):
    p_kernel = kernel_idx[params[0]]
    p_degree = params[1]
    p_gamma = params[2]
    p_coef = params[3]
    p_tol = params[4]
    p_C = params[5]
    p_epsilon = params[6]

    numpy.random.seed(0)
    regr = svm.SVR(kernel=p_kernel, degree=p_degree, gamma=p_gamma, coef0=p_coef, tol=p_tol, C=p_C, epsilon=p_epsilon)
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    return metrics.r2_score(y_test, y_pred)

def fitness(params, x_train, x_test, y_train, y_test):
    try:
        error = test_svm(params, x_train, x_test, y_train, y_test)
        return error
    except:
        print(f"this gave an error: {params}")
        return -100


"""
Params to be optimized:
- type of kernel: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid': [0, 1, 2, 3] Integer
- degree (for 'poly'): [1, 2, ...] Integer
- gamma (for 'rbf', 'poly' and 'sigmoid'): (0 - 1000] Real (maybe exponential mutation)
- coef0 (for 'poly' and 'sigmoid'): [-40 - 40] Real
- tol: (0 - 4] Real
- C: (0 - 1000] Real (maybe exponential mutation)
- epsilon: [0 - 1] Real

"""

def clamp(n, minim, maxim):
    # Limits a number between maxim and minim
    return max(min(n, maxim), minim)

def mutate(params, strength = 0.1, n_params = 7):
    new_params = params.copy()
    for param_idx in range(n_params):
        if random.random() >= strength:
            if param_idx == 0 and random.random() >= 0.5:
                # kernel type, half the probability of changing
                new_params[param_idx] = (new_params[param_idx] + random.randint(0,4))%4
            elif param_idx == 1:
                # degree
                new_params[param_idx] = max(new_params[param_idx] + np.ceil(random.gauss(0, strength*10)), 1)
            elif param_idx == 2 or param_idx == 5:
                # gamma/C
                new_params[param_idx] = 10**(np.log10(params[param_idx]) + random.gauss(0, strength))
                new_params[param_idx] = clamp(new_params[param_idx], 1e-8, 1000)
            elif param_idx == 3:
                # coef
                new_params[param_idx] += random.gauss(0, strength)
                new_params[param_idx] = clamp(new_params[param_idx], -40, 40)
            elif param_idx == 4:
                # tol
                new_params[param_idx] += 10**(np.log10(params[param_idx]) + random.gauss(0, strength))
                new_params[param_idx] = clamp(new_params[param_idx], 1e-8, 4)
            elif param_idx == 6:
                # epsilon
                new_params[param_idx] += 10**(np.log10(params[param_idx]) + random.gauss(0, strength))
                new_params[param_idx] = clamp(new_params[param_idx], 1e-8, 1)
    return new_params


def cross(params1, params2, n_params = 7):
    son1 = params1.copy()
    son2 = params2.copy()
    for i in range(7):
        if random.random() >= 0.5:
            # Swap half of the parameters
            son1[i], son2[i] = son2[i], son1[i]

    return (son1, son2)

def random_params():
    params = [0 for i in range(7)]
    params[0] = random.choice([0,1,2,3])
    params[1] = int(random.uniform(1,8))
    params[2] = 10**random.uniform(-8,0)
    params[3] = random.uniform(-40, 40)
    params[4] = random.uniform(0, 4)
    params[5] = 10**random.uniform(-8,0)
    params[6] = 10**random.uniform(-2,0)

    return params

def lineal(gen, init_str, last_str, max_gen):
    m = (last_str-init_str)/max_gen
    return m * (gen - max_gen)

def genetic_alg(x_train, x_test, y_train, y_test, gen_num = 100, popul_size = 100):
    current_gen = [random_params() for i in range(popul_size)]
    fitness_history = []

    mut_prob = 0.1
    init_str = 0.05
    last_str = 0.001
    for gen in range(gen_num):
        mut_strength = lineal(gen, init_str, last_str, gen_num)

        # Generate offspring
        next_gen = []
        while len(next_gen) <= popul_size:
            parent1 = random.choice(current_gen)
            parent2 = random.choice(current_gen)
            (new_ind1, new_ind2) = cross(parent1, parent2)
            if random.random() <= mut_prob:
                new_ind1 = mutate(new_ind1, mut_strength)
            if random.random() <= mut_prob:
                new_ind2 = mutate(new_ind2, mut_strength)
            next_gen += [new_ind1, new_ind2]

        # Select fittest individuals
        current_gen = sorted(current_gen+next_gen, key=lambda x: fitness(x, x_train, x_test, y_train, y_test), reverse=True)
        current_gen = current_gen[:popul_size]

        fitness_history.append(fitness(current_gen[0], x_train, x_test, y_train, y_test))
        print(f"generation {gen} of {gen_num}")
        print(f"DEBUG: {current_gen[0]}")

    return (current_gen[0], fitness_history)

def get_best_parameters():
    print("Generating data...")
    x_train, x_test, y_train, y_test = init_data(test_size = 0.3)
    print("Searching for the best parameters...")
    indiv, data = genetic_alg(x_train, x_test, y_train, y_test, 200, 100)
    print(indiv)
    print(test_svm(indiv, x_train, x_test, y_train, y_test))
    print(min(data))
    plt.plot(data)
    plt.show()

if __name__ == '__main__':
    get_best_parameters()
