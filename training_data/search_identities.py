import sys
sys.path.append('..')

from Cube import *
from rubikNotation import *
import time
from pathlib import Path

def search_identities(min_moves, max_moves, size=3):
    result = []

    moves = ['U', 'D', 'L', 'R', 'F', 'B']
    moves = [i+modif for i in moves for modif in ('', '\'', '2')]
    opposite = {'R':'L', 'L':'R', 'U':'D', 'D':'U', 'F':'B', 'B':'F', '-':'|'}

    for i in range(min_moves, max_moves+1):
        step_result = []
        DFS_move_search(i, moves, opposite, [], step_result)

        result += step_result
        print(f"With {i} moves: \n{step_result}")

    return result

def DFS_move_search(depth, moves, opposite, visited, result, size=3):
    if depth <= 0:
        cube = Cube(size)
        if cube.doAlgorithm("".join(visited)).isSolved():
            result.append(visited)
    else:
        if len(visited) > 0:
            moves_filtered = [m for m in moves if m[0] != visited[-1][0] and m[0] != opposite[visited[-1][0]]]
        else:
            moves_filtered = moves
        
        for next in moves_filtered:
            DFS_move_search(depth-1, moves, opposite, visited+[next], result)

if __name__ == "__main__":
    a = search_identities(2, 8)
    print(a)
    
    
