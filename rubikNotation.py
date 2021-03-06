import numpy as np

def invertMov(mov):
    """
    str -> str
    OBJ: returns the inverted movement as the one given
    R -> R'
    U2 -> U2
    F' -> F
    """
    result = mov
    if mov[-1] == "'":
        result = mov[0]
    elif mov[-1] != "2":
        result += "'"
    return result

def groupAlg(alg):
    """
    str -> list[str]
    OBJ: groups the movements in an algorithm given by a string
    RL'F2 -> [R,L',F2]
    """
    result=[]
    item=[]
    for i in range(len(alg)):
        item.append(alg[i])
        if len(item) >= 2 or i == len(alg)-1 or (alg[i+1] not in ("'", "2")):
            result.append("".join(item))
            item.clear()

    return result


def invertAlg(alg):
    """
    str -> str
    OBJ: returns the inverse algorithm
    RU'F2 -> F2UR'
    """
    alg = groupAlg(alg)
    stack = []
    result = []

    for mov in alg:
        stack.append(invertMov(mov))
    while len(stack) != 0:
        result.append(stack.pop())

    return "".join(result)

def turnMov(mov, turn):
    """
    str, str -> str
    OBJ: returns the movement given a turn of the entire Cube
    U,x -> B
    F',z' -> F'
    U2,y2 -> D2
    """
    modifier = mov[1:]
    transMap = {}
    if turn[0] == 'x':
        transMap['R'] = 'R'
        transMap['L'] = 'L'
        transMap['M'] = 'M'

        if turn[-1] == "'":
            transMap['U'] = 'F'
            transMap['D'] = 'B'
            transMap['E'] = 'S'
            transMap['F'] = 'D'
            transMap['B'] = 'U'
            transMap['S'] = 'E'
            transMap['x'] = '' #no move
            transMap['y'] = 'z'
            transMap['z'] = 'y\''
        elif turn[-1] == "2":
            transMap['U'] = 'D'
            transMap['D'] = 'U'
            transMap['E'] = 'E\''
            transMap['F'] = 'B'
            transMap['B'] = 'F'
            transMap['S'] = 'S\''
            transMap['x'] = 'x\''
            transMap['y'] = 'y\''
            transMap['z'] = 'z\''
        else:
            transMap['U'] = 'B'
            transMap['D'] = 'F'
            transMap['E'] = 'S'
            transMap['F'] = 'U'
            transMap['B'] = 'D'
            transMap['S'] = 'E'
            transMap['x'] = 'x2'
            transMap['y'] = 'z\''
            transMap['z'] = 'y'

    if turn[0] == 'y':
        transMap['U'] = 'U'
        transMap['D'] = 'D'
        transMap['E'] = 'E'

        if turn[-1] == "'":
            transMap['R'] = 'F'
            transMap['L'] = 'B'
            transMap['M'] = 'M'
            transMap['F'] = 'L'
            transMap['B'] = 'R'
            transMap['S'] = 'S'
            transMap['x'] = 'z\''
            transMap['y'] = ''
            transMap['z'] = 'x\''
        elif turn[-1] == "2":
            transMap['R'] = 'L'
            transMap['L'] = 'R'
            transMap['M'] = 'M\''
            transMap['F'] = 'B'
            transMap['B'] = 'F'
            transMap['S'] = 'S\''
            transMap['x'] = 'x\''
            transMap['y'] = 'y\''
            transMap['z'] = 'z\''
        else:
            transMap['R'] = 'B'
            transMap['L'] = 'F'
            transMap['M'] = 'M'
            transMap['F'] = 'R'
            transMap['B'] = 'L'
            transMap['S'] = 'S'
            transMap['x'] = 'z'
            transMap['y'] = 'y2'
            transMap['z'] = 'x'

    if turn[0] == 'z':
        transMap['F'] = 'F'
        transMap['B'] = 'B'
        transMap['S'] = 'S'

        if turn[-1] == "'":
            transMap['R'] = 'D'
            transMap['L'] = 'U'
            transMap['M'] = 'M'
            transMap['U'] = 'R'
            transMap['D'] = 'L'
            transMap['E'] = 'E'
            transMap['x'] = 'y'
            transMap['y'] = 'x\''
            transMap['z'] = ''
        elif turn[-1] == "2":
            transMap['R'] = 'L'
            transMap['L'] = 'R'
            transMap['M'] = 'M\''
            transMap['U'] = 'D'
            transMap['D'] = 'U'
            transMap['E'] = 'E\''
            transMap['x'] = 'x\''
            transMap['y'] = 'y\''
            transMap['z'] = 'z\''
        else:
            transMap['R'] = 'U'
            transMap['L'] = 'D'
            transMap['M'] = 'M'
            transMap['U'] = 'R'
            transMap['D'] = 'L'
            transMap['E'] = 'E'
            transMap['x'] = 'y\''
            transMap['y'] = 'x'
            transMap['z'] = 'z2'

    if transMap[mov[0]] == '':
        modifier = ''

    return transMap[mov[0]] + modifier

def turnAlg(alg):
    import time
    result = ''
    turns = ('x', 'y', 'z')
    for turn in turns:
        result = alg[:alg.find(turn)]
        while(alg.find(turn) != -1):
            auxAlg = groupAlg(alg[alg.find(turn)+1:])
            for i in auxAlg:
                result += turnMov(i, turn)
            alg = result
    return alg


def transMiddle(alg):
    """
    str -> str
    OBJ: subsitutes parts of an algorithm to get slice movements
    """
    auxStr = alg
    auxStr = auxStr.replace("RL'","Mx")
    auxStr = auxStr.replace("L'R","Mx")
    auxStr = auxStr.replace("R'L","M'x")
    auxStr = auxStr.replace("LR'","M'x")

    auxStr = auxStr.replace("UD'","Ey")
    auxStr = auxStr.replace("D'U","Ey")
    auxStr = auxStr.replace("U'D","E'y")
    auxStr = auxStr.replace("DU'","E'y")

    auxStr = auxStr.replace("FB'","Sz")
    auxStr = auxStr.replace("B'F","Sz")
    auxStr = auxStr.replace("F'B","S'z")
    auxStr = auxStr.replace("BF'","S'z")

    return auxStr
