from Cube import Cube
from rubikNotation import *
import random

# This will get the data to train a Neural network
# that will get an heuristic that will be used
# by the A* search.
# We will not use an optimal heuristic, it
# might overestimate.
class Rubik_train_data:
    def __init__(self, size, depth, max_samples):
        self.cube = Cube(size)
        self.algs = {}
        self.cubes = {}
        self.data = {}
        self.depth = depth
        self.max_samples = max_samples


    def prepare_data(self):
        # generate a large amount of random cubes that are 'depth' moves away from being solved
        moves = ['U', 'D', 'L', 'R', 'F', 'B']
        moves = [i+modif for i in moves for modif in ('', '\'', '2')]
        aux_alg = []
        for i in range(1, self.depth+1):
            self.algs[i] = set()
            self.cubes[i] = set()
            samples = 0
            stall_count = 0
            while(samples < self.max_samples and (stall_count < 10000)):
                aux_alg = [random.choice(moves) for j in range(i)]
                for rep in range(int(i/2)+1):
                    aux_alg = reduxAlg(aux_alg)
                if len(aux_alg) == i and not tuple(aux_alg) in self.algs[i]:
                    self.algs[i].add(tuple(aux_alg))
                    self.cubes[i].add(self.cube.doAlgorithm(aux_alg))
                    samples += 1
                else:
                    stall_count += 1
                #exit if samples > self.max_samples or (samples < len(moves) and i == 1)
        for i in self.cubes:
            self.data[i] = set()
            for j in self.cubes[i]:
                self.data[i].add(tuple(j.get_lin_face_data()))


    def cleanup_data(self):
        # Removes cubes that we believed to be solved in 'i' moves
        # but in reality could be solved faster.
        diff = new_len = orig_len = 0
        for i in reversed(self.cubes):
            for j in reversed(range(1, i)):
                orig_len = len(self.cubes[i])
                self.cubes[i] = self.cubes[i].difference(self.cubes[j])
                new_len = len(self.cubes[i])
                diff += orig_len - new_len
        return diff

    def linearlize_data(self):
        for i in self.cubes:
            self.data[i] = set()
            for j in self.cubes[i]:
                self.data[i].add(tuple(j.get_lin_face_data()))

    def get_net_inputs(self):
        result = []
        if len(self.algs) == 0:
            print("please prepare the input data first")
            return None

        for i in self.data:
            result += self.data[i]
        print(len(result))
        return result

    def get_net_targets(self):
        result = []
        if len(self.algs) == 0:
            print("please prepare the input data first")
            return None
        for i in self.data:
            result += [i] * len(self.data[i])
        print(len(result))
        return result


if __name__ == '__main__':
    rtd = Rubik_train_data(3, 24, 60)
    print(rtd.algs)
