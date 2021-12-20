from Cube import Cube
import numpy as np
import Solver_NN
from Solver_NN.Rubik_solver_NN import *

class Cube_node:
    """
    Struct that has a cube and an
    algorithm that got us to this cube
    """
    def __init__(self, cube, alg):
        self.cube = cube
        self.alg = alg

def eval_nn():


    ## Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load('torchModel.pt').to(device)
    model.eval()

    ## Load training data
    # Inputs
    cube_data = np.loadtxt("NN_input.csv", delimiter=',')
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
    targets1 = np.loadtxt("NN_target.csv", delimiter=',')
    range_rep = np.transpose(np.matlib.repmat(np.arange(1,21), len(targets1), 1))
    targets = np.equal(np.matlib.repmat(targets1, 20, 1), range_rep)
    targets = np.transpose(targets)
    targets = torch.from_numpy(targets).long().to(device)

    t = model(inputs)

    result = torch.argmax(t, dim=1)
    true_targets = torch.argmax(targets, dim=1)

    std = torch.sqrt( torch.sum((result-true_targets)**2)/inputs.shape[0] )

    print(true_targets)
    print(result)



    plt.plot(true_targets.cpu().numpy(), result.cpu().numpy(), 'o')
    plt.plot(numpy.linspace(0, 24, 1000),numpy.linspace(0, 24, 1000))
    plt.show()



class Cube_solver:
    def __init__(self):
        pass

    # We are going to use IDA* (iterative deepening A*)
    def solve_cube(self, cube):
        bound = self.__heuristic(cube)
        path = [Cube_node(cube, [])]
        while True:
            t = self.__search(path, 0, bound)
            if t == -1:
                return (path, bound)
            if t >= 1000:
                return None
            bound = t

    def __search(self, path, g, bound, last_move=''):
        node = path[-1]
        f = g + self.__heuristic(node.cube)
        if f > bound:
            return f
        if node.cube.isSolved():
            return -1
        min = 1000

        # Avilable moves
        moves = ['U', 'D', 'L', 'R', 'F', 'B']

        # Avoid repeating the same type of move
        if last_move in moves:
            moves.remove(last_move)

        # Add al variations of the move
        moves = [i+modif for i in moves for modif in ('', '\'', '2')]

        for succ in moves:
            new_cube = node.cube.turn(succ)
            if new_cube.get_lin_face_data() not in [i.cube.get_lin_face_data() for i in path]:
                path.append(Cube_node(new_cube, node.alg + [succ]))
                t = self.__search(path, g+1, bound, succ[0])
                if t == -1: return -1
                if t < min: min = t
                path.pop()
        return min

    def __heuristic(self, cube):
        return self.__scaled_difference_heuristic(cube)

    def nn_heuristic(self, cube):
        cube_data = np.array(cube.get_lin_face_data())

        colors_in_face = np.array([len(np.unique(i)) for i in np.resize(cube_data, [6, 9])])
        #colors_in_face = torch.stack(colors_in_face)

        inputs = np.concatenate([cube_data, colors_in_face])

        print(inputs)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = torch.load('Heuristic_NN/torchModel.pt').to(device)
        model.eval()

        inputs = torch.tensor(inputs).float().to(device)

        result = model(inputs)
        result = argmax(result)
        print(result)


    def __difference_heuristic(self, cube):
        """
        non admisible heuristic
        """
        lin_data = cube.get_lin_face_data()
        solved = Cube(cube.size).get_lin_face_data()
        value = len(list(filter(lambda x: x[0] != x[1], np.transpose(np.array([lin_data, solved])))))
        return value

    def __scaled_difference_heuristic(self, cube):
        """
        admisible heuristic?
        """
        lin_data = cube.get_lin_face_data()
        solved = Cube(cube.size).get_lin_face_data()
        value = len([i for i in np.transpose(np.array([lin_data, solved])) if i[0] != i[1]])
        return (value * 20)/cube.size**2

if __name__ == '__main__':
    #a = Cube_solver()
    #node, n = a.solve_cube(Cube(3).doAlgorithm(['B','R','U']))
    #print(node[-1].alg)
    #eval_nn()
    a = Cube_solver().nn_heuristic(Cube(3))
