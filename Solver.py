from Cube import Cube
from rubikNotation import *
import numpy as np
from IDA_neural_solver.IDA_neural_solver import *
from Korf_solver.Korf_solver import *
import time
from pathlib import Path
import threading

class Cube_solver_thread(threading.Thread):
    def __init__(self, cube, implementation="IDA_neural"):
        threading.Thread.__init__(self)
        self.cube = cube
        self.solution_found = False
        self.solution = None
        self.implementation = implementation

    def run(self):
        # Set implementation
        if self.implementation == 'Korf':
            solver = Korf_Solver()
        elif self.implementation == 'IDA_neural':
            solver = IDA_Neural_Solver()
        else:
            solver = None

        # If there is no solver return just an U
        if solver is None:
            self.solution = ['U']
        else:
            node, t = solver.solve_cube(self.cube, debug=True)
            self.solution = reduxAlg(node[-1].alg)

        self.solution_found = True


def test_solver(moves = 6, alg = None):
    a = Cube_solver()

    if alg is None:
        c = Cube(3).scramble(moves)
    else:
        c = Cube(3).doAlgorithm(alg)

    node, t = a.solve_cube(c)
    print(node[-1].alg)
    print(reduxAlg(node[-1].alg))
    print(c.toString())
    print(node[-1].cube.toString())
    print("Is it optimal?: ", len(node[-1].alg) <= moves)

def test_solver_time(moves = 6):
    tim = 0
    for i in range(100):
        start = time.time()
        test_solver(moves = moves)
        #test_solver(alg = ['U','R','F','D','F','U\''])
        end = time.time()
        print("time taken:", end-start)
        tim += end-start
    print("avg time", tim/100)


if __name__ == '__main__':
    test_solver(8)
    #test_solver_time(6)

    #NN1: 7.51s
    #NN2: 3.2s
    #NN3: 5.42s
    #NN4: 1.09
    #NN2(symmetry): 0.81


    #eval_nn()
    #a = Cube_solver().nn_heuristic(Cube(3))

    # Real example
    #cube_faces = [0, 5, 5, 1, 0, 4, 2, 5, 3, 2, 4, 3, 4, 1, 1, 0, 0, 1, 0, 2, 2, 0, 2, 3, 4, 2, 5, 5, 0, 4, 2, 3, 2, 4, 5, 5, 3, 3, 1, 1, 4, 3, 1, 5, 3, 0, 0, 1, 3, 5, 1, 4, 4, 2]
    #cube_faces = np.resize(cube_faces, [6,3,3])
