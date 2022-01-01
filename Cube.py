import rubikNotation as rN
import numpy as np
import random
import copy
from Cube3d import Cube3d

class Cube:
    """
               0  0  0
               0  0  0
               0  0  0
      1  1  1  2  2  2  3  3  3  4  4  4
      1  1  1  2  2  2  3  3  3  4  4  4
      1  1  1  2  2  2  3  3  3  4  4  4
               5  5  5
               5  5  5
               5  5  5
    The numbers can mean any combination of 6 colors, one possibility is the following:
    0: white
    1: red
    2: blue
    3: orange
    4: green
    5: yellow
    """
    def __init__(self, size, faces = None):
        """
        Initialize the cube, if the face information is not given,
        the cube will be initialized to a solved state
        """
        self.size = size

        if faces is None:
            self.faces = np.ones([6,3,3], dtype=np.int32)
            for i in range(6):
                self.faces[i] *= i
        else:
            self.faces = np.array(faces)

    def get_lin_face_data(self):
        """
        noInput -> tuple[int]
        linearize the data of the cube, transform a set of 6
        3x3 matrixes to a list with 6*3*3=54 elements
        """
        #lin_face = []
        #for i in range(6):
        #    for j in range(self.size):
        #        lin_face += self.faces[i][j]
        #return lin_face
        return self.faces.flatten()

    def face_to_front(self, face):
        """
        We assume our cube is normalized

        We fix the front facing face, or the blue face in the normalized cube
        """
        color = int(face[1,1])

        aux_cube = Cube(self.size, self.faces)
        transf = []

        # Put our face in position
        orientation_rot = {0:"x'", 1:"y'", 2:"", 3:"y", 4:"y2", 5:"x"}
        transf.append(orientation_rot[color])
        aux_cube = aux_cube.turn(orientation_rot[color])

        transf.append("")

        orientation_found = False
        counter = 0
        rots = {0:"",1:"z",2:"z2",3:"z'"}

        print(self.faces[color,:,:])
        print(face)
        # Orientate the face, check which orientation
        for i in range(4):
            if (self.faces[color,:,:] == np.rot90(face, i)).all():
                transf[1] = rots[i]
                counter += 1

        transf = list(filter(lambda x: x != "", transf))

        return transf, counter

    def normalize_swap_colors(self):
        """
        Only for 3x3 cubes, resets the color of the cube to the original order
        """

        # We take the colors of the centers
        colors = self.faces[:,1,1]

        color_map = {colors[i]: i for i in range(len(colors))}

        new_faces = np.zeros([6, self.size, self.size], dtype=np.uint8)

        for face in range(6):
            for row in range(self.size):
                for color in range(self.size):
                    new_faces[face, row, color] = color_map[self.faces[face, row, color]]

        return Cube(self.size, new_faces)

    def normalize(self, transformation = []):
        """
        We lock in 2 faces that will lock the final rotation of the cube
        The list passed as the argument will be modified to return the transofrmation
        performed to reach the cube.

        We fix the white and red faces to their original positions.

        It works only for the 3x3 cube.
        """
        transformation.clear()
        result = Cube(self.size, self.faces)

        # White face
        white_rots = {0:"", 1:"z", 2:"x", 3:"z'", 4:"x'", 5:"x2"}
        centers = result.faces[:,1,1]
        white_pos = np.where(centers==0)[0][0]
        if white_pos != 0:
            transformation.append(white_rots[white_pos])
            result = result.turn(white_rots[white_pos])

        # Red face
        centers = result.faces[:,1,1]
        red_rots = {1:"", 2:"y", 3:"y2", 4:"y'"}
        red_pos = np.where(centers==1)[0][0]
        if red_pos != 1:
            transformation.append(red_rots[red_pos])
            result = result.turn(red_rots[red_pos])

        return result

    def normalize_corners(self, transformation = []):
        """
        We lock in 2 faces that will lock the final rotation of the cube
        The list passed as the argument will be modified to return the transofrmation
        performed to reach the cube

        We fix the white, red and blue corner to its original positions.

        It works only for the 3x3 cube.
        """

        transformation.clear()
        result = Cube(self.size, self.faces)
        cube3d = Cube3d(self.faces)
        result_pos, orig_colors = cube3d.find_piece_pos([0,1,2])

        pos_layer = result_pos[1:]
        if pos_layer == [0,0]:
            transformation.append("y'")
        elif pos_layer == [0,2]:
            transformation.append("y2")
        elif pos_layer == [2,2]:
            transformation.append("y")

        if result_pos[0] == 2:
            transformation.append("z")

        result = result.doAlgorithm(transformation)

        if result.faces[0,2,0] == 2:
            transformation.append("y'")
            transformation.append("z'")
            result = result.doAlgorithm(["y'", "z'"])
        elif result.faces[0,2,0] == 1:
            transformation.append("z")
            transformation.append("y")
            result = result.doAlgorithm(["z", "y"])

        return result

    def __eq__(self, other):
        """
        Cube -> Bool
        Implements the '==' operation, only if the centers are in the same position
        """
        return (self.normalize().faces == other.normalize().faces).all()

    def __hash__(self):
        """
        noInput -> number
        Implement hash function so this class can be used in sets/hash maps
        """
        return hash(tuple(self.get_lin_face_data()))

    def Uturn(self, times):
        """
        int -> noReturn
        Does a turn a certain number of turns on the upper face of the cube
        """
        times = times%4
        new_faces = self.faces.copy()

        if times != 0:
            new_faces[0] = np.rot90(self.faces[0],(-times)%4)

            # top row of face 1 mapped to 2
            # top row of face 2 mapped to 3
            # top row of face 3 mapped to 4
            # top row of face 4 mapped to 1
            idx_changed = [4,1,2,3]
            for j in range(len(idx_changed)):
                new_faces[idx_changed[j], 0] = self.faces[idx_changed[(j+times)%4], 0]

            #new_faces[idx_changed, 0] = self.faces[np.roll(idx_changed, (j+times%4)), 0]
        return Cube(self.size, new_faces)

    def Xturn(self, times):
        """
        int -> noReturn
        Rotates the cube around the x axis
        """
        times = times%4
        new_faces = self.faces.copy()
        if times != 0:
            # 1 time,  rotate: {0,4}
            # 2 times, rotate: {0,4} + {2,0} = {4,2}
            # 3 times, rotate: {0,4} + {2,0} + {2,5} = {5,4}
            # 4 times, rotate: {0,4} + {2,0} + {2,5} + {5,4} = {}

            new_faces[3] = np.rot90(self.faces[3],(-times)%4)
            new_faces[1] = np.rot90(self.faces[1],(times)%4)

            idx_changed = [0,2,5,4]
            for j in range(len(idx_changed)):
                new_faces[idx_changed[j]] = self.faces[idx_changed[(j+times)%4]]
            #new_faces[idx_changed, 0] = self.faces[np.roll(idx_changed, (j+times%4)), 0]

            new_faces[4] = np.rot90(new_faces[4],2)
            new_faces[idx_changed[(-times-1)%4]] = np.rot90(new_faces[idx_changed[(-times-1)%4]],2)

        return Cube(self.size, new_faces)

    def Yturn(self, times):
        """
        int -> noReturn
        Rotates the cube around the y axis
        """
        times = times%4
        new_faces = self.faces.copy()
        if times != 0:
            new_faces[0] = np.rot90(self.faces[0],(-times)%4)
            new_faces[5] = np.rot90(self.faces[5],(times)%4)

            idx_changed = [4,1,2,3]
            for j in range(len(idx_changed)):
                new_faces[idx_changed[j]] = self.faces[idx_changed[(j+times)%4]]
            #new_faces[idx_changed, 0] = self.faces[np.roll(idx_changed, (j+times%4)), 0]

        return Cube(self.size, new_faces)

    def Zturn(self, times):
        """
        int -> noReturn
        Rotates the cube around the z axis
        """
        times = times%4
        new_faces = self.faces.copy()
        if times != 0:
            new_faces[4] = np.rot90(self.faces[4],(times)%4)
            new_faces[2] = np.rot90(self.faces[2],(-times)%4)

            idx_changed = [0,3,5,1]
            for j in range(len(idx_changed)):
                new_faces[idx_changed[j]] = self.faces[idx_changed[(j-times)%4]]
            #new_faces[idx_changed, 0] = self.faces[np.roll(idx_changed, (j+times%4)), 0]

            new_faces[0] = np.rot90(new_faces[0],(-times))
            new_faces[1] = np.rot90(new_faces[1],(-times))
            new_faces[3] = np.rot90(new_faces[3],(-times))
            new_faces[5] = np.rot90(new_faces[5],(-times))
        return Cube(self.size, new_faces)

    def turn(self, type):
        """
        str -> noReturn
        Does a single turn given the type of the turn
        """
        type = type.upper()

        result = self
        if len(type) > 0:
            if len(type) == 1:
                times = 1
            elif type[1] == '\'':
                times = -1
            elif type[1] == '2':
                times = 2

            if type[0] == 'U':
                result = self.Uturn(times)
            elif type[0] == 'F':
                result = self.Xturn(1).Uturn(times).Xturn(-1)
            elif type[0] == 'D':
                result = self.Xturn(2).Uturn(times).Xturn(2)
            elif type[0] == 'B':
                result = self.Xturn(-1).Uturn(times).Xturn(1)
            elif type[0] == 'R':
                result = self.Zturn(-1).Uturn(times).Zturn(1)
            elif type[0] == 'L':
                result = self.Zturn(1).Uturn(times).Zturn(-1)
            elif type[0] == 'X':
                result = self.Xturn(times)
            elif type[0] == 'Y':
                result = self.Yturn(times)
            elif type[0] == 'Z':
                result = self.Zturn(times)
        else:
            result = self.Uturn(0)

        return result

    def doAlgorithm(self, alg):
        """
        str -> noReturn
        Does a sequence of turns on a cube
        """
        result = self
        grouped = rN.groupAlg(alg)
        for i in grouped:
            result = result.turn(i)
        return result

    def scramble(self, times=20):
        result = []
        moves = ['U', 'D', 'L', 'R', 'F', 'B']
        moves = [i+modif for i in moves for modif in ('', '\'', '2')]
        last_moves = ['--', '--']
        opposite = {'R':'L', 'L':'R', 'U':'D', 'D':'U', 'F':'B', 'B':'F', '-':'|'}
        cube = self

        for i in range(times):
            moves_cleaned = []
            for i in moves:
                aux_move = last_moves[1][0]
                prev_move = last_moves[0][0]
                if i[0] != aux_move:
                    if aux_move == opposite[prev_move]:
                        if i not in (aux_move, opposite[aux_move]):
                            moves_cleaned.append(i)
                    else:
                        moves_cleaned.append(i)

            move = random.choice(moves_cleaned)
            cube = cube.turn(move)

            result.append(move)
            last_moves.pop(0) # dequeue second to last move
            last_moves.append(move) # enqueue last move

        return cube


    def isSolved(self):
        """
        noInput -> bool
        Returns whether the cube is solved(each face has only one color) or not
        """
        solved = True
        i = j = k = 0
        while i < len(self.faces) and solved:
            aux = self.faces[i,0,0]
            j = 0
            while j < len(self.faces[i]) and solved:
                k = 0
                while k < len(self.faces[i,j]) and solved:
                    solved = self.faces[i,j,k] == aux
                    k+=1
                j+=1
            i+=1
        return solved

    def toString(self):
        """
        str -> str
        Returns a text representation of the cube
        """
        result = ""

        result += "size:" + str(self.size) + "\n"

        facesStr = []

        # face 0
        for i in self.faces[0]:
            result += " " * (2*self.size+1)
            for j in i:
                result += str(j) + " "
            result += "\n"

        # faces 1-4 in the same line
        for i in range(self.size):
            facesStr.append([])

        for i in range(1, len(self.faces)-1):
            for j in range(len(self.faces[i])):
                for k in range(len(self.faces[i][j])):
                    facesStr[j] += str(self.faces[i][j][k]) + " "
                facesStr[j] += " "

        for i in facesStr:
            result += "".join(i) + "\n"

        # face 5
        for i in self.faces[5]:
            result += " " * (2*self.size + 1)
            for j in i:
                result += str(j) + " "
            result += "\n"

        return result

    def toStringColor(self):
        """
        str -> str
        Returns a text representation of the cube
        """
        colorMap = {
            0:'\033[48;2;245;245;245m ', # rgb(F5,F5,F5) = (almost)white
            1:'\033[48;2;178;0;0m ',     # rgb(B9,00,00) = red
            2:'\033[48;2;0;69;172m ',    # rgb(00,45,AD) = blue
            3:'\033[48;2;255;89;0m ',    # rgb(FF,59,00) = orange
            4:'\033[48;2;0;155;72m ',    # rgb(00,9B,48) = green
            5:'\033[48;2;213;255;0m '    # rgb(FF,D5,00) = yellow
        }


        result = ""

        #result += "size:" + str(self.size) + "\n"

        facesStr = []

        # face 0
        for i in self.faces[0]:
            result += " " * (2*(self.size+1))
            for j in i:
                result += colorMap[j] + " "
            result += "\033[48;2;0;0;0m\n"

        result += '\n'

        # faces 1-4 in the same line
        for i in range(self.size):
            facesStr.append([])

        for i in range(1, len(self.faces)-1):
            for j in range(len(self.faces[i])):
                for k in range(len(self.faces[i][j])):
                    facesStr[j] += colorMap[self.faces[i][j][k]] + " "
                facesStr[j] += "\033[48;2;0;0;0m  "
        for i in facesStr:
            result += "".join(i) + "\033[48;2;0;0;0m\n"

        result += '\n'

        # face 5
        for i in self.faces[5]:
            result += " " * (2*(self.size + 1))
            for j in i:
                result += colorMap[j] + " "
            result += "\033[48;2;0;0;0m\n"

        return result

    def verify(self):
        """
        noInput -> bool
        Verify that there are 9 colors of each color
        """
        colors = {0:0,1:0,2:0,3:0,4:0,5:0}
        for i in range(6):
            for j in range(self.size):
                for k in range(self.size):
                    colors[self.faces[i][j][k]] += 1

        valid = True
        for i in colors.values():
            valid = valid and i==self.size*self.size
        return valid

if __name__ == '__main__':
    c = Cube(3)
    print(c.verify())
    c.faces[0,0,0] = 1
    print(c.verify())
