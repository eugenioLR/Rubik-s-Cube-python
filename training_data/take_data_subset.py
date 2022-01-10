from pathlib import Path
import numpy as np
import pandas as pd

def remove_last_char(filename):
    with open(filename, 'r') as file:
        data = file.read()[:-1]
    with open(filename, 'w') as file:
        file.write(data)

def reduce_data(data_proportion, input_name, target_name):
    path = str(Path(__file__).resolve().parent) + "/"


    # Take sample of targets
    targets = np.loadtxt(path + "NN_target_full.csv", delimiter=',').astype(np.int8)

    # Ensure we take samples from all distances to the solution
    # Making the data more representative
    cursor = 0
    indices = np.ones(targets.shape)
    for i in range(21):
        amount = np.count_nonzero(targets == i)
        if i not in [0,1,2]:
            take_amount = int(amount*data_proportion)
            idx = np.hstack([np.ones(take_amount), np.zeros(amount - take_amount)])
            np.random.shuffle(idx)
            indices[cursor:cursor+amount] = idx
        cursor += amount
    indices = indices.astype(bool)

    new_targets = targets[indices]
    print("TARGETS")
    print(f"original data matrix size: {targets.shape}")
    print(f"reduced data matrix size: {new_targets.shape}")


    pd.DataFrame(new_targets).to_csv(path + target_name, header=None, index=None, line_terminator = ",")
    remove_last_char(path + target_name)

    # Take sample of inputs
    cube_data = np.loadtxt(path + "NN_input_full.csv", delimiter=',').astype(np.int8)

    new_cube_data = cube_data[indices,:]
    print("INPUTS")
    print(f"original data matrix size: {cube_data.shape}")
    print(f"reduced data matrix size: {new_cube_data.shape}")

    pd.DataFrame(new_cube_data).to_csv(path + input_name, header=None, index=None)


if __name__ == '__main__':
    #print("\nGenerating training data for neural network")
    #reduce_data(0.1, "NN_input_nn.csv", "NN_target_nn.csv")

    #print("\nGenerating training data for svm")
    #reduce_data(0.05, "NN_input_svm.csv", "NN_target_svm.csv")

    print("\nGenerating data for tests")
    reduce_data(0.025, "NN_input_test.csv", "NN_target_test.csv")

    print("\nGenerating data for genetic algorithm")
    reduce_data(0.0002, "NN_input_GA.csv", "NN_target_GA.csv")
