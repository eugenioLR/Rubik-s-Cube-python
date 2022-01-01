import numpy as np
import numpy.random
import torch
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import pickle
import random
from pathlib import Path
import time
from matplotlib import pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def init_data(data_amount = 4000, test_size = 0.3):
    path = str(Path(__file__).resolve().parent) + "/"

    cube_data = np.loadtxt(path + "NN_input.csv", delimiter=',')
    targets = np.loadtxt(path + "NN_target.csv", delimiter=',')

    cube_data = torch.from_numpy(cube_data)

    # Amount of colors in each face
    colors_in_face = []
    for i in range(len(cube_data)):
        colors_in_face.append(torch.tensor([len(torch.unique(i)) for i in torch.reshape(cube_data[i], [6, 9])]))
    colors_in_face = torch.stack(colors_in_face)

    inputs = torch.cat([cube_data, colors_in_face], dim=1)
    inputs = inputs.numpy()

    order = list(range(len(inputs)))
    np.random.shuffle(order)
    order = order[:data_amount]

    inputs = inputs[order].astype(int)
    targets = targets[order].astype(int)

    print(inputs.shape)
    print(targets.shape)

    #x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3)
    return train_test_split(inputs, targets, test_size = test_size)

kernel_idx = {0:'linear', 1:'poly', 2:'rbf', 3:'sigmoid'}
#kernel_idx = {0:'poly', 1:'poly', 2:'poly', 3:'poly'}

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

    start = time.time()
    regr = svm.SVR(kernel=p_kernel, degree=p_degree, gamma=p_gamma, coef0=p_coef, tol=p_tol, C=p_C, epsilon=p_epsilon, shrinking=False, max_iter=10000)
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test).astype(int)
    end = time.time()
    time_spent = end-start
    time_weighted = general_logistic(time_spent, 0.9, 1.1, 0.01, 100)

    #print(time_spent,time_weighted)

    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    #print("Precision:", metrics.precision_score(y_test, y_pred, average='macro'))
    #print("Recall:", metrics.recall_score(y_test, y_pred, average='macro'))

    return metrics.mean_squared_error(y_test, y_pred), time_weighted

#def fitness(params, x_train, x_test, y_train, y_test):
#    error = 0
#    start = time.time()
#    for i in range(20):
#        error += test_svm(params, x_train, x_test, y_train, y_test)
#    end = time.time()
#    time_spent = end-start
#    print(time_spent)
#    return error/100

def fitness(params, x_train, x_test, y_train, y_test):
    error, time = test_svm(params, x_train, x_test, y_train, y_test)
    return error


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
            if param_idx == 0 and random.random() >= strength:
                # kernel type
                new_params[param_idx] = (new_params[param_idx] + random.randint(0,4))%4
            elif param_idx == 1:
                # degree
                new_params[param_idx] = max(new_params[param_idx] + int(random.random()*2 - 2), 1)
            elif param_idx == 2 or param_idx == 5:
                # gamma/C
                new_params[param_idx] += random.choice([1,-1])*10**random.gauss(0, strength*10)
                new_params[param_idx] = clamp(new_params[param_idx], 0.000001, 1000)
            elif param_idx == 3:
                # coef
                new_params[param_idx] += random.gauss(40*strength, strength*4)
                new_params[param_idx] = clamp(new_params[param_idx], -40, 40)
            elif param_idx == 4:
                # tol
                new_params[param_idx] += random.gauss(1*strength, strength*4)
                new_params[param_idx] = clamp(new_params[param_idx], 0.000001, 4)
            #elif param_idx == 6:
                # epsilon
            #    new_params[param_idx] += random.gauss(0.5*strength, strength*4)
            #    new_params[param_idx] = clamp(new_params[param_idx], 0.000001, 1)
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
    params[1] = random.choice([0,1,2,3,4,5,6,7,8,9,10])
    params[2] = 10**random.uniform(-6,3)
    params[3] = random.uniform(-40, 40)
    params[4] = random.uniform(0, 4)
    params[5] = 10**random.uniform(-6,3)
    params[6] = 0.001

    print(params)

    return params

def lineal(gen, init_str, last_str, max_gen):
    m = (last_str-init_str)/max_gen
    return m * (gen - max_gen)

def genetic_alg(x_train, x_test, y_train, y_test, gen_num = 5000, popul_size = 100):
    current_gen = [random_params() for i in range(popul_size)]
    fitness_history = []

    mut_prob = 0.05
    init_str = 0.005
    last_str = 0.0001
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

        current_gen = sorted(current_gen+next_gen, key=lambda x: fitness(x, x_train, x_test, y_train, y_test))
        current_gen = current_gen[:popul_size]

        fitness_history.append(fitness(current_gen[0], x_train, x_test, y_train, y_test))
        print(f"generation {gen} of {gen_num}")

    return (current_gen[0], fitness_history)

def get_best_parameters():
    print("Generating data...")
    x_train, x_test, y_train, y_test = init_data(data_amount = 1000, test_size = 0.3)
    print("Searching for the best parameters...")
    indiv, data = genetic_alg(x_train, x_test, y_train, y_test, 200, 100)
    print(indiv)
    print(test_svm(indiv, x_train, x_test, y_train, y_test))
    print(min(data))
    plt.plot(data)
    plt.show()
