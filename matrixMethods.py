def __turn90(matrix):
    l = len(matrix)
    if l > 1:
        for x in range(l//2):
            for y in range(x, l-x-1):
                turned  = (matrix[y][l-x-1],matrix[l-x-1][l-y-1],matrix[l-y-1][x],matrix[x][y])
                (matrix[x][y],matrix[y][l-x-1],matrix[l-x-1][l-y-1],matrix[l-y-1][x]) = turned
    return matrix

def __turn180(matrix):
    l = len(matrix)
    if l > 1:
        for x in range(l//2):
            for y in range(x, l-x-1):
                turned  = (matrix[y][l-x-1],matrix[l-x-1][l-y-1],matrix[l-y-1][x],matrix[x][y])
                (matrix[l-y-1][x],matrix[x][y],matrix[y][l-x-1],matrix[l-x-1][l-y-1]) = turned
    return matrix

def __turn270(matrix):
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
    if times%4 == 1:
        matrix = __turn90(matrix)
    if times%4 == 2:
        matrix = __turn180(matrix)
    if times%4 == 3:
        matrix = __turn270(matrix)
    return matrix
