import numpy as np
import time
from pathlib import Path
import threading

from ..Cube import Cube
from ..rubikNotation import *

# from .IDA_SVMSolver import *
from .IDA_NNSolver import *
from .IDA_Solver import *
# from .Korf_Solver import *

class Cube_solver_thread(threading.Thread):
    def __init__(self, cube, implementation="IDA*"):
        threading.Thread.__init__(self)
        self.cube = cube
        self.solution_found = False
        self.solution = None
        self.transform = None
        self.implementation = implementation

    def run(self):
        # Set implementation
        if self.implementation == 'Korf':
            print("Not implemented yet")
            solver = None
        elif self.implementation == 'IDA*-NN':
            solver = IDA_NNSolver()
        elif self.implementation == 'IDA*-SVM':
            solver = IDA_SVMSolver()
        elif self.implementation == 'IDA*':
            solver = IDA_Solver()
        elif self.implementation == 'RNN':
            print("Not implemented yet")
            solver = None
        else:
            solver = None

        # If there is no solver return just an U
        if solver is None:
            self.solution = ['Incorrect cube solver']
        else:
            node, t = solver.solve_cube(self.cube, debug=True)
            self.solution = reduxAlg(node[-1].alg)
            self.transform = t
            self.nodes_generated = solver.nodes_generated

        self.solution_found = True


def test_solver(moves, implementation, alg = None):
    if alg is None:
        c = Cube(3).scramble(moves)
    else:
        c = Cube(3).doAlgorithm(alg)
    solver = Cube_solver_thread(c, implementation)

    start = time.time()
    solver.start()
    solver.join(120)
    end = time.time()

    if solver.solution_found:
        print(solver.solution)
        print(c.toStringColor())
        print("Is it optimal?: ", len(solver.solution) <= moves)
        print(f"Time taken: {end-start}s")
        print(f"Nodes generated: {solver.nodes_generated}")
    else:
        print("Took way too much")
    return end - start

def test_solver_time(moves, implementation):
    tim = 0
    for i in range(100):
        print(f"\nSolving cube {i} using {implementation}")
        tim += test_solver(moves, implementation)
    print("avg time", tim/100)


if __name__ == '__main__':
    #test_solver(10, 'IDA*-NN', ["B2", "U'", "D2", "B", "F", "R'", "F'"])
    test_solver_time(5, 'IDA*')
    # time spent for 8 moves: 6681.240522623062 = 1 hour 51 minutes 21 seconds
