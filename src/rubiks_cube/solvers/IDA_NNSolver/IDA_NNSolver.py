import time
import random
from pathlib import Path
import torch
from .Rubik_heuristic_NN import NeuralNetwork
from ...Cube3d import Cube3d
from ...Cube import Cube
from ...rubikNotation import *
from ..Solver import Solver


class Cube_node:
    """
    Struct that has a cube and an
    algorithm that got us to this cube
    """
    def __init__(self, cube, alg):
        self.cube = cube
        self.alg = alg

class IDA_NNSolver(Solver):
    def __init__(self):
        path = str(Path(__file__).resolve().parent) + "/"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(path + '3x3HeuristicModel_config1.pt').to(self.device)
        self.model.eval()
        self.min = 100
        self.nodes_generated = 0

    # We are going to use IDA* (iterative deepening A*)
    def solve_cube(self, cube, debug = False):
        self.nodes_generated = 0
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

        # Experimental
        random.shuffle(moves)

        for succ in moves:
            new_cube = node.cube.turn(succ)
            #if new_cube not in [i.cube for i in path]:
            if self.already_checked(new_cube, path) and len(reduxAlg(node.alg + [succ])) == len(node.alg + [succ]):
                self.nodes_generated += 1
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
        perc = [0.9, 0.1]
        neural_heuristic = self.__nn_heuristic(cube)
        piece_distance_heuristic = self.__manhattan_dist(cube)
        return int(perc[0]*neural_heuristic + perc[1]*piece_distance_heuristic)
        #return mi(neural_heuristic, piece_distance_heuristic)

    def __nn_heuristic(self, cube):
        cube_data = np.array(cube.normalize().get_lin_face_data())
        colors_in_face = np.array([len(np.unique(i)) for i in np.resize(cube_data, [6, 9])])
        # Uncomment if the net accepts 60 inputs
        inputs = np.concatenate([cube_data, colors_in_face])
        inputs = torch.tensor([inputs]).float().to(self.device)

        #inputs = torch.tensor(cube_data).float().to(self.device)
        result = self.model(inputs)

        # USED FOR 'ONE HOT' ENCODING
        #result = result.softmax(dim=1).argmax().to('cpu')
        result = result.argmax().to('cpu')


        return int(result)

    def __manhattan_dist(self, cube):
        c3d = Cube3d(cube.faces)
        corner_distance, edge_distance = c3d.dist_to_solution()

        return int(max(corner_distance/4, edge_distance/4))
