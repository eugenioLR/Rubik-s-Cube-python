from Cube import Cube

def start():
    size = 0
    try:
        size = int(input("introduce the size of the cube: "))
    except:
        size = 3

    return Cube(size)

def performMovements(scramble, cube):
    algorithm = scramble
    while(algorithm != "EXIT"):
        cube.doAlgorithm(algorithm)
        print(cube.toString())
        algorithm = input("introduce the algorithm you want to execute ('exit' to exit, 'reset' to reset):").upper()
        if algorithm == "RESET":
            cube = Cube(cube.size)
            algorithm = scramble

def menu():
    cube = start()

    running = True
    while running:
        print("------------------------------------------")
        print("0-Change size of the cube.")
        print("1-Solve a cube.")
        print("2-Transform algorithm.")
        print("e-Exit")
        print("------------------------------------------")
        option = input("Select an optin:")
        if option == "0":
            cube = start()
        elif option == "1":
            alg = input("insert the initial scramble(leave empty to do nothing): ")
            performMovements(alg, cube)
        elif option == "2":
            pass
        elif option == "e":
            running = False

if __name__ == "__main__":
    menu()
