
#? ---------------
#? --- IMPORTS ---
#? ---------------
import cv2
import math
import numpy as np
import operator
import os
import glob
import random
import copy
import time
from tqdm import tqdm

import tkinter as tk
from tkinter import filedialog

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import load_model

#? -----------------
#? --- FUNCTIONS ---
#? -----------------
def preprocess_image(image):
    # Make image grayscale to remove colors
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur image so lines stand out more
    processed_img = cv2.GaussianBlur(processed_img.copy(), (9, 9), 3)

    # Use tresholding to differentiate background and foreground
    processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert black and white
    processed_img = cv2.bitwise_not(processed_img, processed_img)

    return processed_img

def prepare_image(image):
    width = image.shape[0]
    height = image.shape[1]

    center = (width/2, height/2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 3)
    tresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    result = cv2.bitwise_not(tresh, tresh)
    
    # Remove small noise
    kernel = np.ones((3, 3), np.uint8) 
    result = cv2.erode(result, kernel)
    result = cv2.dilate(result, kernel)

    contours, _ = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    dist = []

    closest_pnt = []
    for contour in contours:
        points = []
        for point in contour:
            xa, ya = point[0]
            xb, yb = center
            distance = math.sqrt((xb - xa)**2 + (yb - ya)**2)
            points.append(distance)
        closest_pnt.append(min(points))

    for contour in contours:
        if cv2.contourArea(contour) < 250:
            cv2.fillPoly(result, pts=[contour], color=(0,0,0))
            continue

        points = []
        for point in contour:
            xa, ya = point[0]
            xb, yb = center
            distance = math.sqrt((xb - xa)**2 + (yb - ya)**2)
            points.append(distance)

        if min(points) > min(closest_pnt):
            cv2.fillPoly(result, pts=[contour], color=(0,0,0))
            continue
    
    return result

def find_corners(image):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))

    return [contour[top_left][0], contour[top_right][0], contour[bottom_right][0], contour[bottom_left][0]]

def four_point_transform(pts, image):
    width = image.shape[0]
    height = image.shape[1]

    # Corner coördinates in original image
    pts1 = np.float32(pts)

    # Destination coördinates
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]]) 
        
    # Apply Perspective Transform Algorithm 
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    result = cv2.warpPerspective(image, matrix, (width, height))

    return result

def predict_numbers(image, model):
    sq_width = int(image.shape[0] / 9)
    sq_height = int(image.shape[1] / 9)

    # Loop trough all numbers, do some preprocessing and predict the number
    sudoku = [[0 for i in range(9)] for j in range(9)]
    for i in range(9):
        # row = []
        for j in range(9):
            crop_img = image[i*sq_height:i*sq_height+sq_height, j*sq_width:j*sq_width+sq_width]
            crop_img = cv2.resize(crop_img, (100, 100))
            crop_img = prepare_image(crop_img)

            predictions = model.predict_proba(np.array(crop_img).reshape(1, 100, 100, 1))[0]
            prediction = np.argmax(predictions, axis=0)
            confidence = predictions[prediction]

            if confidence > 0.70:
                if validate(j, i, prediction, sudoku):
                    sudoku[i][j] = prediction

    return sudoku

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


#? --------------------
#? --- MAIN PROGRAM ---
#? --------------------
if __name__ == '__main__':
    # Load Keras model
    model = load_model('models/number_classifier_v1.0.hdf5')

    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames()

    for file in file_paths:
        # Read image of sudoku and resize
        image = cv2.imread(file)
        image = cv2.resize(image, (720, 720))

        # print('Starting...')
        start = time.time()

        # Some preprocessing to eliminate everything but the sudoku from the picture
        preprocessed_image = preprocess_image(image)
        corners = find_corners(preprocessed_image)
        result = four_point_transform(corners, image)

        # Loop trough every row and predict numbers
        sudoku = predict_numbers(result, model)
        end = time.time()
        print(f'\nScanning sudoku took: {end-start}')

        print_sudoku(sudoku)
        print('\n-----------------------')
        print('\nsolving sudoku ...\n')

        start = time.time()
        # Solving predicted sudoku
        board = copy.deepcopy(sudoku)
        if solve(board) == False:
            print('SUDOKU CANT BE SOLVED!')

        end = time.time()
        print(f'\nSolving sudoku took: {end-start}')
        print_sudoku(board)
    


        cv2.imshow('Result', result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()