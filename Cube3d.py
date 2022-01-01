class Cube_piece:
    def __init__(self, pos, colors):
        self.pos = pos
        self.colors = colors

    def dist_to_solution(self, piece_pos):
        pass

class Cube_edge(Cube_piece):
    def __init__(self, colors):
        pos = -1

        colors_s = sorted(colors)

        if colors_s[0] == 0:
            if colors_s[1] == 4:
                pos = [0,0,1]
            elif colors_s[1] == 1:
                pos = [0,1,0]
            elif colors_s[1] == 3:
                pos = [0,1,2]
            elif colors_s[1] == 2:
                pos = [0,2,1]
            elif colors_s[1] == 5:
                [2,1,0]

        elif colors_s[0] == 1:
            if colors_s[1] == 2:
                pos = [1,2,0]
            elif colors_s[1] == 4:
                pos = [1,0,0]
            elif colors_s[1] == 5:
                pos = [2,1,0]


        elif colors_s[0] == 2:
            if colors_s[1] == 3:
                pos = [1,2,2]
            elif colors_s[1] == 4:
                pos = [1,2,0]
            elif colors_s[1] == 5:
                pos = [2,0,1]


        elif colors_s[0] == 3:
            if colors_s[1] == 4:
                pos = [1,0,2]
            elif colors_s[1] == 5:
                pos = [2,1,2]
        elif colors_s[0] == 4:
            pos = [2,2,1]

        super(Cube_edge, self).__init__(pos, colors)

        if pos == -1:
            print("something went wrong")
            print(colors)

    def dist_to_solution(self, piece_pos):
        dists = []
        for i in range(len(piece_pos)):
            dists.append(abs(self.pos[i] - piece_pos[i]))
        return sum(dists)


class Cube_corner(Cube_piece):
    def __init__(self, colors):
        pos = -1

        colors_s = sorted(colors)

        if colors_s[0] == 0:
            if colors_s[1] == 1:
                if colors_s[2] == 4:
                    pos = [0,0,0]
                elif colors_s[2] == 2:
                    pos = [0,0,2]
            elif colors_s[1] == 2:
                pos = [0,2,2]
            elif colors_s[1] == 3:
                pos = [0,2,0]
        elif colors_s[0] == 1:
            if colors_s[1] == 2:
                pos = [2,0,2]
            elif colors_s[0] == 1:
                pos = [2,0,0]
        elif colors_s[0] == 2:
            pos = [2,2,2]
        elif colors_s[0] == 3:
            pos = [2,2,0]


        super(Cube_corner, self).__init__(pos, colors)

        if pos == -1:
            print("something went wrong")
            print(colors)

    def dist_to_solution(self, piece_pos):
        dists = []
        for i in range(len(piece_pos)):
            dists.append(abs(self.pos[i] - piece_pos[i]))
        return sum(dists)

class Cube3d():
    def __init__(self, faces):
        # We don't store the centers, we assume the cube is normalized

        # tensor of rank 3
        self.pieces = [[[None for i in range(3)] for j in range(3)] for k in range(3)]

        # Top face corners
        self.pieces[0][0][0] = Cube_corner([faces[0,0,0],faces[1,0,0],faces[4,0,2]])
        self.pieces[0][2][0] = Cube_corner([faces[0,0,2],faces[3,0,2],faces[4,0,0]])
        self.pieces[0][0][2] = Cube_corner([faces[0,2,0],faces[1,0,2],faces[2,0,0]])
        self.pieces[0][2][2] = Cube_corner([faces[0,2,2],faces[2,0,2],faces[3,0,0]])

        # Top face edges
        self.pieces[0][0][1] = Cube_edge([faces[0,0,1],faces[4,0,1]])
        self.pieces[0][1][0] = Cube_edge([faces[0,1,0],faces[1,0,1]])
        self.pieces[0][1][2] = Cube_edge([faces[0,1,2],faces[3,0,1]])
        self.pieces[0][2][1] = Cube_edge([faces[0,2,1],faces[2,0,1]])

        # Middle row edges
        self.pieces[1][0][0] = Cube_edge([faces[1,1,0],faces[4,1,2]])
        self.pieces[1][2][0] = Cube_edge([faces[1,1,2],faces[2,1,0]])
        self.pieces[1][2][2] = Cube_edge([faces[2,1,2],faces[3,1,0]])
        self.pieces[1][0][2] = Cube_edge([faces[3,1,2],faces[4,1,0]])


        # Bottom face corners
        self.pieces[2][0][2] = Cube_corner([faces[1,2,2],faces[2,2,0],faces[5,0,0]])
        self.pieces[2][2][2] = Cube_corner([faces[2,2,2],faces[3,2,0],faces[5,0,2]])
        self.pieces[2][0][0] = Cube_corner([faces[1,2,0],faces[4,2,2],faces[5,2,0]])
        self.pieces[2][2][0] = Cube_corner([faces[3,2,2],faces[4,2,0],faces[5,2,2]])

        # Bottom face edges
        self.pieces[2][0][1] = Cube_edge([faces[2,2,1],faces[5,0,1]])
        self.pieces[2][1][0] = Cube_edge([faces[1,2,1],faces[5,1,0]])
        self.pieces[2][1][2] = Cube_edge([faces[3,2,1],faces[5,1,2]])
        self.pieces[2][2][1] = Cube_edge([faces[4,2,1],faces[5,2,1]])

    def dist_to_solution(self):
        edge_distance = 0
        corner_distance = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if self.pieces[i][j][k] is not None:
                        if isinstance(self.pieces[i][j][k], Cube_corner):
                            corner_distance += self.pieces[i][j][k].dist_to_solution([i,j,k])
                        else:
                            edge_distance += self.pieces[i][j][k].dist_to_solution([i,j,k])
        return corner_distance, edge_distance

    def find_piece_pos(self, colors):
        result_pos = []
        orig_colors = []
        found = False
        i = 0
        while i < 3 and not found:
            j = 0
            while j < 3 and not found:
                k = 0
                while k < 3 and not found:
                    if self.pieces[i][j][k] is not None:
                        result_pos = [i,j,k]
                        orig_colors = self.pieces[i][j][k].colors
                        found = sorted(self.pieces[i][j][k].colors) == colors
                    k += 1
                j += 1
            i += 1

        if not found:
            print("something went wrong")

        return result_pos, orig_colors
