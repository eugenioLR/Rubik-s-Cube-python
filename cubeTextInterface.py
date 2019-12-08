from Cube import Cube

size = 0

try:
    size = int(input("introduce the size of the cube: "))
except:
    size = 3

cube = Cube(size)

algorithm = ""
while(algorithm != "EXIT"):
    print(cube.toString())
    algorithm = input("introduce the algorithm you want to execute ('exit' to exit):").upper()
    cube.doAlgorithm(algorithm)
