import os
from copy import copy
from Cube import Cube
from rubikNotation import *

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
            algorithm = input("introduce the algorithm you want to execute ('exit' to exit, 'reset' to reset):").upper()
            if algorithm == "RESET":
                cube = Cube(cube.size)
                algorithm = scramble
    
    def transformAlgorithm(self, alg):
        algorithm_original = groupAlg(alg)
        algorithm = copy(algorithm_original)
        option = ""
        while option != "EXIT":
            self.clearScreen()
            print("Current algorithm:", *algorithm)
            option = input("introduce the algorithm you want to execute ('exit' to exit, 'reset' to reset):").upper()
            if option == "INVERT":
                algorithm = invertAlg(algorithm)




    def menu(self):
        cube = self.start()

        running = True
        while running:
            self.clearScreen()
            print("------------------------------------------")
            print("0-Change size of the cube.")
            print("1-Solve a cube.")
            print("2-Transform algorithm.")
            print("e-Exit")
            print("------------------------------------------")
            option = input("Select an option:")
            if option == "0":
                cube = self.start()
            elif option == "1":
                alg = input("insert the initial scramble(leave empty to do nothing): ")
                self.performMovements(alg, cube)
            elif option == "2":
                alg = input("insert the initial scramble(leave empty to do nothing): ")
                self.transformAlgorithm(alg)
            elif option == "e":
                running = False

    def clearScreen(self):
        try:
            os.system("cls")
            os.system("clear")
        except:
            pass

if __name__ == "__main__":
    menu = Menu()
    menu.menu()
