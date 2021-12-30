from Cube import Cube
from rubikNotation import *
import numpy as np
import time
from pathlib import Path
import threading


class Cube_node:
    """
    Struct that has a cube and an
    algorithm that got us to this cube
    """
    def __init__(self, cube, alg):
        self.cube = cube
        self.alg = alg

class Cube_piece:
    def __init__(self, pos, colors):
        self.pos = pos
        self.colors = colors

    def dist_to_solution(self, piece_pos):
        pass

class Korf_Solver:
    # https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf
    def __init__(self):
        self.min = 100

    # We are going to use IDA* (iterative deepening A*)
    def solve_cube(self, cube, debug = False):
        solution = []
        transf = []
        cube = cube.normalize(transf)

        bound = self.__heuristic(cube)
        path = [Cube_node(cube, [])]
        t = 0
        while t >= 0 and t <= self.min:
            t = self.__IDAstar_search(path, 0, bound)
            bound = t
            if debug:
                print("DEBUG: bound found", bound)

        if t == -1:
            return (path, transf)
        if t >= self.min:
            return None

    def __IDAstar_search(self, path, g, bound, last_move=''):
        node = path[-1]
        f = g + self.__heuristic(node.cube)
        if f > bound:
            return f
        if node.cube.isSolved():
            return -1
        min = self.min

        # Avilable moves
        moves = ['U', 'D', 'L', 'R', 'F', 'B']

        # Avoid repeating the same type of move
        if last_move in moves:
            moves.remove(last_move)

        # Add al variations of the move
        moves = [i+modif for i in moves for modif in ('', '\'', '2')]

        for succ in moves:
            new_cube = node.cube.turn(succ)
            #if new_cube not in [i.cube for i in path]:
            if self.already_checked(new_cube, path) and len(reduxAlg(node.alg + [succ])) == len(node.alg + [succ]):
                path.append(Cube_node(new_cube, node.alg + [succ]))
                t = self.__IDAstar_search(path, g+1, bound, succ[0])
                if t == -1:
                    return -1
                if t < min:
                    min = t
                path.pop()
        return min

    def already_checked(self, cube, path):
        # Check for symetries in the cube set
        cubes = [i.cube for i in path]

        new_cubes = []
        for rot1 in ['', 'x', 'x2', 'x\'', 'z', 'z\'']:
            for rot2 in ['', 'y', 'y2', 'y\'']:
                new_cubes.append(cube.turn(rot1).turn(rot2).normalize_swap_colors())

        found = False
        i = 0
        while not found and i < len(new_cubes):
            found = new_cubes[i] not in [j.cube for j in path]
            i += 1

        return found

    def __heuristic(self, cube):
        return self.__manhattan_dist(cube)

    def __manhattan_dist(self, cube):
        c3d = Cube3d(cube.faces)
        corner_distance, edge_distance = c3d.dist_to_solution()

        return int(max(corner_distance/4, edge_distance/4))

def test_solver(moves = 6, alg = None):
    a = Korf_Solver()

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
