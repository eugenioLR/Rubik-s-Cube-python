import numpy as np

def invertMov(mov):
    """
    O(1)
    """
    result = mov
    if mov[-1] == "'":
        result = mov[0]
    elif mov[-1] != "2":
        result += "'"
    return result

def groupAlg(alg):
    """
    O(n)
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
    O(n)
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
    O(1)
    """
    transMap = {}
    if turn[0] == 'x':
        transMap['R'] = 'R'
        transMap['L'] = 'L'
        transMap['M'] = 'M'

        if turn[-1] == "'":
            transMap['U'] = 'B'
            transMap['D'] = 'F'
            transMap['E'] = 'S'
            transMap['F'] = 'U'
            transMap['B'] = 'D'
            transMap['S'] = 'E'
        elif turn[-1] == "2":
            transMap['U'] = 'D'
            transMap['D'] = 'U'
            transMap['E'] = 'E\''
            transMap['F'] = 'B'
            transMap['B'] = 'F'
            transMap['S'] = 'S\''
        else:
            transMap['U'] = 'F'
            transMap['D'] = 'B'
            transMap['E'] = 'S'
            transMap['F'] = 'D'
            transMap['B'] = 'U'
            transMap['S'] = 'E'

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
        elif turn[-1] == "2":
            transMap['R'] = 'L'
            transMap['L'] = 'R'
            transMap['M'] = 'M\''
            transMap['F'] = 'B'
            transMap['B'] = 'F'
            transMap['S'] = 'S\''
        else:
            transMap['R'] = 'B'
            transMap['L'] = 'F'
            transMap['M'] = 'M'
            transMap['F'] = 'R'
            transMap['B'] = 'L'
            transMap['S'] = 'S'

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
        elif turn[-1] == "2":
            transMap['R'] = 'L'
            transMap['L'] = 'R'
            transMap['M'] = 'M\''
            transMap['U'] = 'D'
            transMap['D'] = 'U'
            transMap['E'] = 'E\''
        else:
            transMap['R'] = 'U'
            transMap['L'] = 'D'
            transMap['M'] = 'M'
            transMap['U'] = 'R'
            transMap['D'] = 'L'
            transMap['E'] = 'E'
    return transMap[mov[0]] + mov[1:]

def turnAlg(alg):
    """
    O(n^2)
    """
    transStack = []
    movStack = []

    #reverse algorithm into auxAlg
    revStack = []
    for i in alg:
        revStack.append(i)
    auxAlg = ' '
    while len(revStack) != 0:
        auxAlg += revStack.pop()

    #split by x, y and z
    cursor=0
    for i in auxAlg:
        if i in (' ', 'x','y','z'):
            transStack.append(i)
            movStack.append('')
            cursor+=1
        else:
            movStack[cursor] += i

    toTrans = []
    for i in movStack:
        for j in i:
            toTrans.append(groupAlg(j))

    #transform each movement


    #reverse auxAlg
    for i in auxAlg:
        revStack.append(i)
    auxAlg = ''
    while len(revStack) != 0:
        auxAlg += revStack.pop()

    return alg

def transMiddle(alg):
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




def main():
    strIn = input("introduce an algorithm to invert: ")
    strIn = strIn.replace(" ", "")
    strIn = strIn.replace("(", "")
    strIn = strIn.replace(")", "")
    print(invertAlg(strIn))
    #for turn in ('x', 'y', 'z'):
    #    for mov in ('R','L','U','D','F','B'):
    #        print(turn + mov + "-" + turnMov(mov,turn))
    #print(turnAlg("RxRyRz"))

if __name__=="__main__":
    main()
