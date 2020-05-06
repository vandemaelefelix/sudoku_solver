import numpy as np
import random
import copy 

sudoku = [
    [
        [5, 3, 0,   0, 7, 0,   0, 0, 0],
        [6, 0, 0,   1, 9, 5,   0, 0, 0],
        [0, 9, 8,   0, 0, 0,   0, 6, 0],

        [8, 0, 0,   0, 6, 0,   0, 0, 3],
        [4, 0, 0,   8, 0, 3,   0, 0, 1],
        [7, 0, 0,   0, 2, 0,   0, 0, 6],

        [0, 6, 0,   0, 0, 0,   2, 8, 0],
        [0, 0, 0,   4, 1, 9,   0, 0, 5],
        [0, 0, 0,   0, 8, 0,   0, 7, 9]
    ],
    [
        [0, 0, 0,   0, 9, 0,   6, 0, 5],
        [0, 8, 0,   1, 0, 6,   0, 0, 0],
        [0, 4, 0,   0, 0, 3,   0, 1, 0],

        [6, 0, 0,   0, 0, 5,   0, 0, 7],
        [0, 0, 0,   0, 0, 0,   0, 0, 0],
        [0, 0, 1,   6, 7, 8,   3, 4, 0],

        [0, 0, 0,   0, 0, 0,   0, 0, 1],
        [0, 0, 0,   3, 0, 4,   0, 0, 0],
        [2, 0, 0,   0, 0, 0,   4, 0, 0]
    ],
    [
        [0, 0, 0,   0, 0, 0,   0, 0, 0],
        [0, 0, 0,   0, 0, 0,   0, 0, 0],
        [0, 0, 0,   0, 0, 0,   0, 0, 0],

        [0, 0, 0,   0, 0, 0,   0, 0, 0],
        [0, 0, 0,   0, 0, 0,   0, 0, 0],
        [0, 0, 0,   0, 0, 0,   0, 0, 0],

        [0, 0, 0,   0, 0, 0,   0, 0, 0],
        [0, 0, 0,   0, 0, 0,   0, 0, 0],
        [0, 0, 0,   0, 0, 0,   0, 0, 0]
    ]
]

def print_sudoku(sudoku):
    print('.   0 1 2     3 4 5     6 7 8')
    print()
    for i in range(len(sudoku)):
        if i % 3 == 0 and i != 0:
            print('    -------------------------')

        print(i, end='   ')
        
        for j in range(len(sudoku[i])):
            if j % 3 == 0 and j != 0:
                print(' | ', end=" ")
            print(sudoku[i][j], end=" ")
        print()

def find_empty(sudoku):
    for row_i, row in enumerate(sudoku):
        for val_i, val in enumerate(row):
            if val == 0:
                return val_i, row_i
    return None

def validate(x, y, value, sudoku):

    # Check row
    for i in sudoku[y]:
        if i == value:
            return False

    # Check column
    for row in sudoku:
        if value == row[x]:
            return False
    
    # Find out in what square it is located
    square_x = x // 3
    square_y = y // 3

    # Check square
    for i in sudoku[square_y*3:square_y*3+3]:
        if value in i[square_x*3:square_x*3+3]:
            return False
        
    return True

def solve(sudoku):
    empty = find_empty(sudoku)

    # If no empty cells are found, the sudoku board is solved
    if not empty:
        return True

    # Get position empty cell
    x, y = empty
    
    # Try numbers 1-9 in the cell
    for i in range(1, 10):
        # If the number is valid place it in the board and check if the board is solved
        if validate(x, y, i, sudoku):
            sudoku[y][x] = i

            if solve(sudoku):
                return True

            sudoku[y][x] = 0

    return False

board = copy.deepcopy(sudoku[2])
solve(board)
print_sudoku(board)