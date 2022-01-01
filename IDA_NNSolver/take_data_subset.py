from pathlib import Path
import numpy as np
import pandas as pd

def reduce_data(data_proportion, input_name, target_name):
    path = str(Path(__file__).resolve().parent) + "/"


    # Take sample of targets
    targets = np.loadtxt(path + "NN_target_full.csv", delimiter=',').astype(np.int8)

    # Ensure we take samples from all distances to the solution
    # Making the data more representative
    pointer = 0
    indices = np.ones(targets.shape)
    for i in range(21):
        amount = np.count_nonzero(targets == i)
        if i not in [0,1,2,3,4]:
            take_amount = int(amount*data_proportion)
            idx = np.hstack([np.ones(take_amount), np.zeros(amount - take_amount)])
            np.random.shuffle(idx)
            indices[pointer:pointer+amount] = idx
        pointer += amount
    indices = indices.astype(bool)

    new_targets = targets[indices]
    print(f"original data matrix size: {targets.shape}")
    print(f"reduced data matrix size: {new_targets.shape}")


    pd.DataFrame(new_targets).to_csv(path + target_name, header=None, index=None)

    # Take sample of inputs
    cube_data = np.loadtxt(path + "NN_input_full.csv", delimiter=',').astype(np.int8)

    new_cube_data = cube_data[indices,:]
    print(f"original data matrix size: {cube_data.shape}")
    print(f"reduced data matrix size: {new_cube_data.shape}")

    pd.DataFrame(new_cube_data).to_csv(path + input_name, header=None, index=None)


if __name__ == '__main__':
    reduce_data(0.1, "NN_input.csv", "NN_target.csv")
    reduce_data(0.05, "NN_input_test.csv", "NN_target_test.csv")
