import os
from copy import copy
from .Cube import Cube
from .rubikNotation import *

class Menu:
    def __init__(self):
        self.clearScreen()
        self.colorCheck()

    def start(self):
        self.clearScreen()
        self.size = 0
        try:
            self.size = int(input("introduce the size of the cube: "))
        except:
            self.size = 3

        return Cube(self.size)

    def colorCheck(self):
        print("\033[48;2;255;0;0mtest\033[48;2;0;0;0m")
        self.color = input("Is the text above red?(Y/N)").upper() == 'Y'
        self.clearScreen()

    def performMovements(self, scramble, cube):
        algorithm = scramble
        while algorithm != "EXIT":
            self.clearScreen()
            cube = cube.doAlgorithm(algorithm)
            if cube.isSolved():
                print("Solution found.")

            if self.color :
                print(cube.toStringColor())
            else:
                print(cube.toString())
            algorithm = input("Introduce the algorithm you want to execute ('exit' to exit, 'reset' to reset):").upper()
            if algorithm == "RESET":
                cube = Cube(cube.size)
                algorithm = scramble
    
    def transformAlgorithm(self, alg):
        algorithm_original = groupAlg(alg)
        algorithm = copy(algorithm_original)
        option = ""
        running = True
        # while option != "EXIT":
        while running:
            self.clearScreen()
            print("------------------------------------------")
            print("n-Input new algorithm.")
            print("ar-Append moves (from the right).")
            print("al-Append moves (from the left).")
            print("s-Simplify algorithm.")
            print("i-Invert move.")
            print("r-Reset.")
            print("e-Exit.")
            print("------------------------------------------")
            print("Current algorithm:", *algorithm)
            print("------------------------------------------")
            match input("Select an option:"):
                case "n":
                    new_algorithm = input("insert the new algorithm: ")
                    algorithm = groupAlg(new_algorithm)
                case "ar":
                    new_algorithm = input("insert new moves: ")
                    algorithm += groupAlg(new_algorithm)
                case "al":
                    new_algorithm = input("insert new moves: ")
                    algorithm = groupAlg(new_algorithm) + algorithm
                case "s":
                    algorithm = reduxAlg(algorithm)
                case "i":
                    algorithm = invertAlg(algorithm)
                case "r":
                    algorithm = algorithm_original
                case "e":
                    running = False

    def menu(self):
        cube = self.start()

        running = True
        while running:
            self.clearScreen()
            print("------------------------------------------")
            print("0-Change size of the cube.")
            print("1-Solve a cube.")
            print("2-Transform algorithm.")
            print("e-Exit.")
            print("------------------------------------------")
            match input("Select an option:"):
                case "0":
                    cube = self.start()
                case "1":
                    alg = input("Insert the initial scramble (leave empty to do nothing): ")
                    self.performMovements(alg, cube)
                case "2":
                    alg = input("Insert the initial algorithm: ")
                    self.transformAlgorithm(alg)
                case "e":
                    running = False
                case _:
                    pass

    def clearScreen(self):
        try:
            os.system("clear")
        except:
            try:
                os.system("cls")
            except:
                pass

def main():
    menu = Menu()
    menu.menu()

if __name__ == "__main__":
    main()
