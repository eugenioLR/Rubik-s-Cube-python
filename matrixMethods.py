import numpy as np
import copy

def turnM(matrix,times):
    times = times%4
    return np.rot90(matrix, times)
