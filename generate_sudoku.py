import numpy as np
import random
import copy
import tqdm

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

    if not empty:
        return True

    x, y = empty

    tested = []
    while len(tested) < 9:
        number = random.randint(1, 9)
        if validate(x, y, number, sudoku):
            sudoku[y][x] = number

            if solve(sudoku):
                return True

            sudoku[y][x] = 0
        
        tested.append(number)

    return False

def generate(size=(9, 9), difficulty=60):
    original = [[0 for i in range(size[0])] for i in range(size[1])]
    solved = copy.deepcopy(original)
    solve(solved)

    original = copy.deepcopy(solved)
    number_empty = 0
    while number_empty < difficulty:
        print('.', end='')
        x, y = (random.randint(0, 8), random.randint(0, 8))
        # if original[y][x] == 0:
        #     continue
        # print(x, y)
        org_copy = copy.deepcopy(original)
        org_copy[y][x] = 0
        if not solve(org_copy):
            continue
        else:
            if original[y][x] == 0:
                continue
            number_empty += 1
            original[y][x] = 0

    print(f'\nNumber empty: {number_empty}')
    return original, solved

print_sudoku(generate()[0])