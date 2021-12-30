from Cube import Cube
from rubikNotation import *
from IDA_neural_solver.Rubik_heuristic_NN import *
import time
from pathlib import Path


class Cube_node:
    """
    Struct that has a cube and an
    algorithm that got us to this cube
    """
    def __init__(self, cube, alg):
        self.cube = cube
        self.alg = alg

class IDA_Neural_Solver:
    def __init__(self):
        path = str(Path(__file__).resolve().parent) + "/"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(path + '3x3HeuristicModel2.pt.bak').to(self.device)
        self.model.eval()
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
        return self.__nn_heuristic(cube)

    def __nn_heuristic(self, cube):
        cube_data = np.array(cube.normalize().get_lin_face_data())
        colors_in_face = np.array([len(np.unique(i)) for i in np.resize(cube_data, [6, 9])])
        inputs = np.concatenate([cube_data, colors_in_face])
        inputs = torch.tensor([inputs]).float().to(self.device)
        result = self.model(inputs)
        # USED FOR 'ONE HOT' ENCODING
        result = result.softmax(dim=1).argmax().to('cpu')

        return int(result)

def eval_nn():

    ## Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load('Heuristic_NN/3x3HeuristicModel.pt').to(device)
    model.eval()

    ## Load training data
    # Inputs
    cube_data = np.loadtxt("Heuristic_NN/NN_input.csv", delimiter=',')
    cube_data = torch.from_numpy(cube_data)

    # Amount of colors in each face
    colors_in_face = []
    for i in range(len(cube_data)):
        colors_in_face.append(torch.tensor([len(torch.unique(i)) for i in torch.reshape(cube_data[i], [6, 9])]))
    colors_in_face = torch.stack(colors_in_face)

    # Corners
    indices = torch.tensor([0,2,6,8])
    indices = torch.cat([indices+(9*i) for i in range(6)])
    corners = cube_data[:,indices]

    inputs = torch.cat([cube_data, colors_in_face], dim=1).float().to(device)
    #inputs = torch.cat([corners, colors_in_face], dim=1).float()
    #inputs = colors_in_face.float()

    # Targets
    targets = np.loadtxt("Heuristic_NN/NN_target.csv", delimiter=',')
    range_rep = np.transpose(np.matlib.repmat(np.arange(1,21), len(targets1), 1))
    targets = np.equal(np.matlib.repmat(targets1, 20, 1), range_rep)
    targets = np.transpose(targets)
    targets = torch.from_numpy(targets).long().to(device)

    t = model(inputs)

    result = torch.argmax(t, dim=1)
    true_targets = torch.argmax(targets, dim=1)
    result = torch.round(t.reshape(t.shape[0], -1)).cpu().detach()
    true_targets = targets.reshape(targets.shape[0], -1).cpu().detach()

    #std = torch.sqrt(torch.sum((result-true_targets)**2)/inputs.shape[0] )

    print(true_targets)
    print(result)

    plt.plot(true_targets.numpy(), result.numpy(), 'o')
    plt.plot(numpy.linspace(0, 24, 1000),numpy.linspace(0, 24, 1000))
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.show()

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
