import argparse
from rubiks_cube.Cube import *
from rubiks_cube.rubikNotation import *
import time
from pathlib import Path

from joblib import Parallel, delayed 

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

    def get_random_alg(self, times):
        result = []
        moves = ['U', 'D', 'L', 'R', 'F', 'B']
        moves = [i+modif for i in moves for modif in ('', '\'', '2')]
        last_moves = ['--', '--']
        opposite = {'R':'L', 'L':'R', 'U':'D', 'D':'U', 'F':'B', 'B':'F', '-':'|'}

        for _ in range(times):
            moves_cleaned = []
            for j in moves:
                aux_move = last_moves[1][0]
                prev_move = last_moves[0][0]
                if j[0] != aux_move:
                    if aux_move == opposite[prev_move]:
                        if j not in (aux_move, opposite[aux_move]):
                            moves_cleaned.append(j)
                    else:
                        moves_cleaned.append(j)

            move = random.choice(moves_cleaned)

            result.append(move)
            last_moves.pop(0) # dequeue second to last move
            last_moves.append(move) # enqueue last move
        return result

    def prepare_data(self, debug = False):

        # generate a large amount of random cubes that are 'depth' moves away from being solved
        self.cubes[0] = [self.cube]
        self.algs[0] = [""]

        def prepare_data_depth(i):
            algs = list()
            cubes = list()
            cube_set = set()
            samples = 0
            stall_count = 0
            
            total_samples = 1

            if i > 6:
                max_stall = self.max_stall_unknown
            else:
                max_stall = self.max_stall
            
            j = 0

            while samples < self.max_samples and stall_count < max_stall:
                aux_alg = self.get_random_alg(i)
                new_cube = self.cube.doAlgorithm(aux_alg)
                if not new_cube in cube_set:
                    cubes.append(new_cube)
                    cube_set.add(new_cube)
                    algs.append("".join(aux_alg))
                    samples += 1
                elif self.distance_pos[i] != 0 and len(cubes) >= self.distance_pos[i]:
                    stall_count += 1
                j += 1
                if debug and j%20000 == 0 and j != 0:
                    print(f"DEBUG: PROGRESS, iteration {j} with {len(cubes)} data points at depth {i}")

            total_samples += len(cubes)
            if debug:
                print(f"DEBUG: DONE, depth {i} reached, generated {len(cubes)} datapoints")
            
            return (i, total_samples, cubes, algs)

        res = Parallel(n_jobs=self.depth)(delayed(prepare_data_depth)(i) for i in range(1, self.depth+1))
        
        total_samples = 0
        for val in res:
            total_samples += val[1]
            self.cubes[val[0]] = val[2]
            self.algs[val[0]] = val[3]

        return total_samples


    def cleanup_data(self, debug = False):
        # Removes cubes that were believed to be solved in 'i' moves
        # but could be solved in less moves.
        diff = new_len = orig_len = 0

        for i in range(len(self.cubes)-1, 0, -1):
            for j in range(i-1, -1, -1):
                orig_len = len(self.cubes[i])
                self.cubes[i] = self.cubes[i].difference(self.cubes[j])
                new_len = len(self.cubes[i])
                diff += orig_len - new_len

                if debug and orig_len - new_len > 0:
                    amount = orig_len - new_len
                    print(f"DEBUG: removed {amount} cube{'s' if amount > 1 else ''} solvable in {j} moves that {'were' if amount > 1 else 'was'} generated with {i} moves")
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

    def flatten_data(self):
        for i in self.cubes:
            self.data[i] = set()
            for j in self.cubes[i]:
                self.data[i].add(tuple(j.get_flat_face_data()))
            self.cubes[i] = None

    def get_net_inputs(self):
        result = []

        for i in self.data:
            result += self.data[i]
        
        print(len(result))
        return result

    def get_net_targets(self):
        result = []

        for i in self.data:
            result += [i] * len(self.data[i])
        print(len(result))
        return result
    
    def get_moves(self):
        result = []

        for i in self.algs:
            result += self.algs[i]
        print(len(result))
        return result

def gen_data(max_depth, max_states, path, debug):
    data = Rubik_train_data(3, max_depth, max_states)

    cubes = data.prepare_data(debug=debug)
    print(f"Generated {cubes} data points")

    # data_purged = data.cleanup_data(debug=debug)
    # print(f"Got rid of {data_purged} data points")

    #data_expanded = data.expand_data()
    #print(f"Generated {data_expanded} new data points")

    #data_purged = data.cleanup_data()
    #print(f"Got rid of {data_purged} data points after expanding the data")

    data.flatten_data()    

    inputs = data.get_net_inputs()
    
    with open(path + "NN_input_full.csv", "w") as file_in:
        for i in inputs:
            file_in.write(",".join([str(j) for j in i]))
            file_in.write("\n")
    
    targets = data.get_net_targets()
    with open(path + "NN_target_full.csv", "w") as file_targ:
        file_targ.write(str(targets[0]))
        for i in targets[1:]:
            file_targ.write(",")
            file_targ.write(str(i))
    
    moves = data.get_moves()
    with open(path + "NN_moves_full.csv", "w") as file_targ:
        for i in moves:
            file_targ.write(i)
            file_targ.write("\n")

def generate_data(max_depth, max_states, path, debug):
    # Go crazy with the amount of data, you can take a subset of the total data after that
    start = time.time()
    gen_data(max_depth, max_states, path=path, debug=debug)
    end = time.time()
    print(f"Time spent: {end-start}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default="./data/")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-m", "--max_depth", type=int, default=20)
    parser.add_argument("-s", "--max_states", type=int, default=1000000)
    args = parser.parse_args()

    generate_data(args.max_depth, args.max_states, args.path, args.debug)

if __name__ == '__main__':
    main()
