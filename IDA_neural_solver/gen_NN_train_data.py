import sys
sys.path.append("..")

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
        # To avoid inifite loops, it can be done better...
        self.max_stall = 10000
        self.max_stall_unknown = max_samples
        # check the last table in: https://cube20.org/ and https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf
        # number of cube states by distance
        self.distance_pos = {
            0: 1,
            1: 18,
            2: 243,
            3: 3240,
            4: 43239,
            5: 574908,
            6: 7618438,
            # the rest are too high to even reach, we wont take them into account
        }
        for i in range(7, 22):
            self.distance_pos[i] = 0



    def prepare_data(self, debug = False):

        # generate a large amount of random cubes that are 'depth' moves away from being solved
        moves = ['U', 'D', 'L', 'R', 'F', 'B']
        moves = [i+modif for i in moves for modif in ('', '\'', '2')]
        aux_alg = []
        self.cubes[0] = set([self.cube])
        self.algs[0] = [['']]
        total_samples = 1
        for i in range(1, self.depth+1):
            self.algs[i] = set()
            self.cubes[i] = set()
            samples = 0
            stall_count = 0
            max_stall = self.max_stall
            if i > 6:
                max_stall = self.max_stall_unknown
            j = 0
            while(samples < self.max_samples and stall_count < max_stall):
                aux_alg = [random.choice(moves) for j in range(i)]
                aux_alg = reduxAlg(aux_alg)
                if len(aux_alg) == i and not tuple(aux_alg) in self.algs[i]:
                    self.algs[i].add(tuple(aux_alg))
                    self.cubes[i].add(self.cube.doAlgorithm(aux_alg))
                    samples += 1
                elif self.distance_pos[i] != 0:
                    if len(self.cubes[i]) >= self.distance_pos[i]:
                        stall_count += 1
                j += 1
                if debug and j%100000 == 0 and j != 0:
                    print(f"DEBUG: iteration {j} with {len(self.cubes[i])} data points at depth {i}")

            total_samples += len(self.cubes[i])
            if debug:
                print(f"DEBUG: depth {i} reached, generated {len(self.cubes[i])} datapoints")
        return total_samples


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

    def expand_data(self):
        # Rotates all the cubes and adds them to the hashmap
        rotations = ['x', 'y', '']
        rotations = [i+modif for i in rotations for modif in ('', '\'', '2')]
        count = 0

        for i in self.cubes:
            for rot1 in ['', 'x', 'x2', 'x\'', 'z', 'z\'']:
                for rot2 in ['', 'y', 'y2', 'y\'']:
                    if not(rot1 == '' and rot1 == ''):
                        aux_set = set()
                        for j in self.cubes[i]:
                            aux_set.add(j.turn(rot1).turn(rot2))
                        self.cubes[i] = self.cubes[i].union(aux_set)
            count += len(self.cubes[i])
        return count

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

def gen_data():
    data = Rubik_train_data(3, 20, 75000)

    cubes = data.prepare_data(debug=True)
    print(f"Generated {cubes} data points")

    data_purged = data.cleanup_data()
    print(f"Got rid of {data_purged} data points")

    #data_expanded = data.expand_data()
    #print(f"Generated {data_expanded} new data points")

    #data_purged = data.cleanup_data()
    #print(f"Got rid of {data_purged} data points after expanding the data")

    data.linearlize_data()

    inputs = data.get_net_inputs()
    targets = data.get_net_targets()

    with open("NN_input.csv", "w") as file_in:
        for i in inputs:
            file_in.write(",".join([str(j) for j in i]))
            file_in.write("\n")

    with open("NN_target.csv", "w") as file_targ:
        file_targ.write(str(targets[0]))
        for i in targets[1:]:
            file_targ.write(",")
            file_targ.write(str(i))

if __name__ == '__main__':
    gen_data()
