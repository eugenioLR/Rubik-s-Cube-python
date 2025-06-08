from Cube3d import Cube3d
from Cube import Cube
from rubikNotation import *
import time
import random
from pathlib import Path

class Node:
    total_sims = 0

    def __init__(self, cube, alg):
        self.cube = cube
        self.alg = alg
        self.value = -1
        self.children = []
        self.parent = None
        self.sims = 0
        self.visited = False

def monte_carlo_tree_search():
    root = Node(Cube(3), [])

    solved = False
    while not solved:
        leaf = traverse(root)
        sim_res = rollout(leaf)
        backpropagate(leaf, sim_res)
        solved = sim_res == 0
    
    return best_child(root)

def traverse(node):

    pass

def rollout(node):
    while node != None:
        node = random.choice(node.children)
    return node

# def rollout_policy(node):
#     return random.choice(node.children)

def backpropagate(node, result):
    if node.parent is not None:
        node.value = update_val(node, result)
        backpropagate(node.parent, result+1)

def update_val(node, result, c=1.4142):
    s = node.value + c * np.sqrt(np.log(Node.total_sims)/self.sims)
    return s
