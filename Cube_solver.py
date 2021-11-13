class Cube_node:
    def __init__(self, cube, alg):
        self.cube = cube
        self.alg = alg


class Cube_solver:
    def __init__(self):
        pass

    # We are going to use IDA* (iterative deepening A*)
    def solve_cube(self, cube):
        bound = self.heuristic(cube)
        path = [Cube_node(cube, [])]
        while True:
            t = self.__search(path, 0, bound)
            if t == -1:
                return (path, bound)
            if t >= 1000:
                return None
            bound = t

    def __search(self, path, g, bound):
        node = path[-1]
        f = g + self.heuristic(node.cube)
        if f < bound: return f
        if node.isSolved(): return -1
        min = 1000

        moves = ['U', 'D', 'L', 'R', 'F', 'B']
        moves = [i+modif for i in moves for modif in ('', '\'', '2')]
        for succ in moves:
            new_cube = node.cube.turn(succ)
            if new_cube.get_lin_face_data() not in [i.cube.get_lin_face_data() for i in path]:
                path.push(Cube_node(new_cube, node.alg + [succ]))
                t = self.__search(path, g+1, bound)
                if t == -1: return -1
                if t < min: min = t
                path.pop()
        return min
