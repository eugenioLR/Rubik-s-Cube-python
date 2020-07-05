import os
from Cube import Cube

COLOR = True

def start():
    clearScreen()
    colorCheck()
    size = 0
    try:
        size = int(input("introduce the size of the cube: "))
    except:
        size = 3

    return Cube(size)

def colorCheck():
    global COLOR
    print("\033[48;2;255;0;0mtest\033[48;2;0;0;0m")
    COLOR = input("Is the text above red?(Y/N)").upper() == 'Y'
    clearScreen()

def performMovements(scramble, cube):
    global COLOR
    algorithm = scramble
    while(algorithm != "EXIT"):
        clearScreen()
        cube.doAlgorithm(algorithm)
        if cube.isSolved():
            print("Solution found.")
        if COLOR:
            print(cube.toStringColor())
        else:
            print(cube.toString())
        algorithm = input("introduce the algorithm you want to execute ('exit' to exit, 'reset' to reset):").upper()
        if algorithm == "RESET":
            cube = Cube(cube.size)
            algorithm = scramble

def menu():
    cube = start()

    running = True
    while running:
        clearScreen()
        print("------------------------------------------")
        print("0-Change size of the cube.")
        print("1-Solve a cube.")
        print("2-Transform algorithm.")
        print("e-Exit")
        print("------------------------------------------")
        option = input("Select an option:")
        if option == "0":
            cube = start()
        elif option == "1":
            alg = input("insert the initial scramble(leave empty to do nothing): ")
            performMovements(alg, cube)
        elif option == "2":
            pass
        elif option == "e":
            running = False

def clearScreen():
    try:
        os.system("cls")
        os.system("clear")
    except:
        pass

if __name__ == "__main__":
    menu()
