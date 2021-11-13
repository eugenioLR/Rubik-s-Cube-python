import matrixMethods as mM
import rubikNotation as rN
import copy

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
    """
    def __init__(self, size, faces = None):
        self.size = size
        self.faces = faces
        if self.faces is None:
            self.faces = []
            for i in range(6):
                self.faces.append([])
                for j in range(size):
                    self.faces[i].append([])
                    for k in range(size):
                        self.faces[i][j].append(i)

    def get_lin_face_data(self):
        lin_face = []
        for i in range(6):
            for j in range(self.size):
                lin_face += self.faces[i][j]
        return lin_face

    def __eq__(self, other):
        # Implements the '==' operation
        # It is not yet independent of rotation
        aux_cube = other
        turns = ['x']
        for i in range(6):
            for j in range(self.size):
                if self.faces[i][j] != other.faces[i][j]:
                    return False
        return True

    def __hash__(self):
        # Implement hash function so this class can be used in sets/hash maps
        return hash(tuple(self.get_lin_face_data()))

    def Uturn(self, times):
        """
        int -> noReturn
        OBJ: does a turn a certain number of turns on the upper face of the cube
        """
        times = times%4
        new_faces = copy.deepcopy(self.faces)

        if times != 0:
            new_faces[0] = mM.turnM(self.faces[0],-times)

            # top row of face 1 mapped to 2
            # top row of face 2 mapped to 3
            # top row of face 3 mapped to 4
            # top row of face 4 mapped to 1
            idx_changed = [4,1,2,3]
            for j in range(len(idx_changed)):
                new_faces[idx_changed[j]][0] = self.faces[idx_changed[(j+times)%4]][0]
        return Cube(self.size, new_faces)

    def Xturn(self, times):
        """
        int -> noReturn
        OBJ: rotates the cube around the x axis
        """
        times = times%4
        new_faces = copy.deepcopy(self.faces)
        if times != 0:
            # 1 time,  rotate: {0,4}
            # 2 times, rotate: {0,4} + {2,0} = {4,2}
            # 3 times, rotate: {0,4} + {2,0} + {2,5} = {5,4}
            # 4 times, rotate: {0,4} + {2,0} + {2,5} + {5,4} = {}

            new_faces[3] = mM.turnM(self.faces[3],-times)
            new_faces[1] = mM.turnM(self.faces[1],times)

            idx_changed = [0,2,5,4]
            for j in range(len(idx_changed)):
                new_faces[idx_changed[j]] = self.faces[idx_changed[(j+times)%4]]

            new_faces[4] = mM.turnM(new_faces[4],2)
            new_faces[idx_changed[(-times-1)%4]] = mM.turnM(new_faces[idx_changed[(-times-1)%4]],2)

        return Cube(self.size, new_faces)

    def Yturn(self, times):
        """
        int -> noReturn
        OBJ: rotates the cube around the y axis
        """
        times = times%4
        new_faces = copy.deepcopy(self.faces)
        if times != 0:
            new_faces[0] = mM.turnM(self.faces[0],-times)
            new_faces[5] = mM.turnM(self.faces[5],times)

            idx_changed = [4,1,2,3]
            for j in range(len(idx_changed)):
                new_faces[idx_changed[j]] = self.faces[idx_changed[(j+times)%4]]
        return Cube(self.size, new_faces)

    def Zturn(self, times):
        """
        int -> noReturn
        OBJ: rotates the cube around the z axis
        """
        times = times%4
        new_faces = copy.deepcopy(self.faces)
        if times != 0:
            new_faces[4] = mM.turnM(self.faces[4],times)
            new_faces[2] = mM.turnM(self.faces[2],-times)

            idx_changed = [0,3,5,1]
            for j in range(len(idx_changed)):
                new_faces[idx_changed[j]] = self.faces[idx_changed[(j-times)%4]]

            new_faces[0] = mM.turnM(new_faces[0],-times)
            new_faces[1] = mM.turnM(new_faces[1],-times)
            new_faces[3] = mM.turnM(new_faces[3],-times)
            new_faces[5] = mM.turnM(new_faces[5],-times)
        return Cube(self.size, new_faces)

    def turn(self, type):
        """
        str -> noReturn
        OBJ: does a single turn given the type of the turn
        """
        result = self
        if len(type) == 1:
            times = 1
        elif type[1] == '\'':
            times = -1
        elif type[1] == '2':
            times = 2

        if type[0].upper() == 'U':
            result = self.Uturn(times)
        elif type[0].upper() == 'F':
            result = self.Xturn(1).Uturn(times).Xturn(-1)
        elif type[0].upper() == 'D':
            result = self.Xturn(2).Uturn(times).Xturn(2)
        elif type[0].upper() == 'B':
            result = self.Xturn(-1).Uturn(times).Xturn(1)
        elif type[0].upper() == 'R':
            result = self.Zturn(-1).Uturn(times).Zturn(1)
        elif type[0].upper() == 'L':
            result = self.Zturn(1).Uturn(times).Zturn(-1)
        elif type[0].upper() == 'X':
            result = self.Xturn(times)
        elif type[0].upper() == 'Y':
            result = self.Yturn(times)
        elif type[0].upper() == 'Z':
            result = self.Zturn(times)
        return result

    def doAlgorithm(self, alg):
        """
        str -> noReturn
        OBJ: does a sequence of turns on a cube
        """
        result = self
        grouped = rN.groupAlg(alg)
        for i in grouped:
            result = result.turn(i)
        return result

    def isSolved(self):
        """
        noInput -> bool
        OBJ: returns whether the cube is solved(each face has only one color) or not
        """
        solved = True
        i = j = k = 0
        while i < len(self.faces) and solved:
            aux = self.faces[i][0][0]
            j = 0
            while j < len(self.faces[i]) and solved:
                k = 0
                while k < len(self.faces[i][j]) and solved:
                    solved = self.faces[i][j][k] == aux
                    k+=1
                j+=1
            i+=1
        return solved

    def toString(self):
        """
        str -> str
        OBJ: returns a text representation of the cube
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
        OBJ: returns a text representation of the cube
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
