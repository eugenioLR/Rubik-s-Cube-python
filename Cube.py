import matrixMethods as mM
import rubikNotation as rN

class Cube:
    """
            0 0 0
            0 0 0
            0 0 0
      1 1 1 2 2 2 3 3 3 4 4 4
      1 1 1 2 2 2 3 3 3 4 4 4
      1 1 1 2 2 2 3 3 3 4 4 4
            5 5 5
            5 5 5
            5 5 5
    """
    def __init__(self, size):
        self.size = size
        self.faces = []
        for i in range(6):
            self.faces.append([])
            for j in range(size):
                self.faces[i].append([])
                for k in range(size):
                    self.faces[i][j].append(i)

    def Uturn(self, times):
        """
        int -> noReturn
        OBJ: does a turn a certain number of turns on the upper face of the cube
        """
        self.faces[0] = mM.turnM(self.faces[0],-times)
        for i in range(times%4):
            turned = (self.faces[1][0],self.faces[2][0],self.faces[3][0],self.faces[4][0])
            (self.faces[4][0],self.faces[1][0],self.faces[2][0],self.faces[3][0]) = turned

    def Xturn(self, times):
        """
        int -> noReturn
        OBJ: rotates the cube around the x axis
        """
        self.faces[3] = mM.turnM(self.faces[3],-times)
        self.faces[1] = mM.turnM(self.faces[1],times)

        for i in range(times%4):
            self.faces[4] = mM.turnM(self.faces[4],2)
            self.faces[0] = mM.turnM(self.faces[0],2)

            turned = (self.faces[2],self.faces[5],self.faces[4],self.faces[0])
            (self.faces[0],self.faces[2],self.faces[5],self.faces[4]) = turned

    def Yturn(self, times):
        """
        int -> noReturn
        OBJ: rotates the cube around the y axis
        """
        self.faces[0] = mM.turnM(self.faces[0],-times)
        self.faces[5] = mM.turnM(self.faces[5],times)
        for i in range(times%4):

            turned = (self.faces[1],self.faces[2],self.faces[3],self.faces[4])
            (self.faces[4],self.faces[1],self.faces[2],self.faces[3]) = turned

    def Zturn(self, times):
        """
        int -> noReturn
        OBJ: rotates the cube around the z axis
        """
        self.faces[4] = mM.turnM(self.faces[4],-times)
        self.faces[2] = mM.turnM(self.faces[2],times)
        for i in range(times%4):
            self.faces[0] = mM.turnM(self.faces[0],1)
            self.faces[1] = mM.turnM(self.faces[1],1)
            self.faces[3] = mM.turnM(self.faces[3],1)
            self.faces[5] = mM.turnM(self.faces[5],1)
            turned = (self.faces[3],self.faces[5],self.faces[1],self.faces[0])
            (self.faces[0],self.faces[3],self.faces[5],self.faces[1]) = turned

    def turn(self, type):
        """
        str -> noReturn
        OBJ: does a single turn given the type of the turn
        """
        if len(type) == 1:
            times = 1
        elif type[1] == '\'':
            times = -1
        elif type[1] == '2':
            times = 2

        if type[0].upper() == 'U':
            self.Uturn(times)
        if type[0].upper() == 'F':
            self.Xturn(1)
            self.Uturn(times)
            self.Xturn(-1)
        if type[0].upper() == 'D':
            self.Xturn(2)
            self.Uturn(times)
            self.Xturn(2)
        if type[0].upper() == 'B':
            self.Xturn(-1)
            self.Uturn(times)
            self.Xturn(1)
        if type[0].upper() == 'R':
            self.Zturn(1)
            self.Uturn(times)
            self.Zturn(-1)
        if type[0].upper() == 'L':
            self.Zturn(-1)
            self.Uturn(times)
            self.Zturn(1)
        if type[0].upper() == 'X':
            self.Xturn(times)
        if type[0].upper() == 'Y':
            self.Yturn(times)
        if type[0].upper() == 'Z':
            self.Zturn(times)

    def doAlgorithm(self, alg):
        """
        str -> noReturn
        OBJ: does a sequence of turns on a cube
        """
        grouped = rN.groupAlg(alg)
        for i in grouped:
            self.turn(i)

    def isSolved(self):
        """
        noInput -> bool
        OBJ: returns whether the cube is solved(each face has only one color) or not
        """
        solved = True
        i = j = k = 0
        aux = -1
        while i < self.size and solved:
            while j < len(self.faces[i]) and solved:
                while k < len(self.faces[i][j]) and solved:
                    if aux != -1:
                        solved = self.faces[i][j][k] == aux
                    aux = self.faces[i][j][k]
                    k+=1
                aux = -1
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
