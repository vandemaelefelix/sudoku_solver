import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import glob, os
import time
import operator



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

    index = 0
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contour = sorted(contours, key=cv2.contourArea, reverse=True)[index]

    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    print(approx)

    while len(approx) > 4:
        index += 1
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contour = sorted(contours, key=cv2.contourArea, reverse=True)[index]

        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    

    cv2.drawContours(image, [contour], 0, (0, 255, 0), 5)
    

    corners = find_corners(preprocessed_image)

    result = four_point_transform(corners, image)

    cv2.imshow('sudoku', image)
    key = cv2.waitKey(0)

    if key == 27:
        break