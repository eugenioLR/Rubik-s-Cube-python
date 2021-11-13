import copy

def __turn90(mat):
    matrix = copy.deepcopy(mat)
    l = len(matrix)
    if l > 1:
        for x in range(l//2):
            for y in range(x, l-x-1):
                turned  = (matrix[y][l-x-1],matrix[l-x-1][l-y-1],matrix[l-y-1][x],matrix[x][y])
                (matrix[x][y],matrix[y][l-x-1],matrix[l-x-1][l-y-1],matrix[l-y-1][x]) = turned
    return matrix

def __turn180(mat):
    matrix = copy.deepcopy(mat)
    l = len(matrix)
    if l > 1:
        for x in range(l//2):
            for y in range(x, l-x-1):
                turned  = (matrix[y][l-x-1],matrix[l-x-1][l-y-1],matrix[l-y-1][x],matrix[x][y])
                (matrix[l-y-1][x],matrix[x][y],matrix[y][l-x-1],matrix[l-x-1][l-y-1]) = turned
    return matrix

def __turn270(mat):
    matrix = copy.deepcopy(mat)
    l = len(matrix)
    if l > 1:
        for x in range(l//2):
            for y in range(x, l-x-1):
                turned  =(matrix[x][y],matrix[y][l-x-1],matrix[l-x-1][l-y-1],matrix[l-y-1][x])
                (matrix[y][l-x-1],matrix[l-x-1][l-y-1],matrix[l-y-1][x],matrix[x][y]) = turned
    return matrix

def printM(matrix, endM="\n"):
    for i in matrix:
        for j in i:
            print(j, end="")
        print()
    print(end=endM)

def turnM(matrix,times):
    times = times%4
    if times == 1:
        matrix = __turn90(matrix)
    elif times == 2:
        matrix = __turn180(matrix)
    elif times == 3:
        matrix = __turn270(matrix)
    return matrix
