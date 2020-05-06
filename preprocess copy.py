import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def preprocess(image):
    width = image.shape[0]
    height = image.shape[1]

    center = (width/2, height/2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 3)
    tresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    result = cv2.bitwise_not(tresh, tresh)

    contours, _ = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8) 
    result = cv2.erode(result, kernel)
    result = cv2.dilate(result, kernel)

    dist = []

    closest_pnt = []
    for contour in contours:
        points = []
        for point in contour:
            xa, ya = point[0]
            xb, yb = center
            distance = math.sqrt((xb - xa)**2 + (yb - ya)**2)
            points.append(distance)
            print(f'Distance: {distance}')
        closest_pnt.append(min(points))

    for contour in contours:
        if cv2.contourArea(contour) < 50:
            cv2.fillPoly(result, pts=[contour], color=(0,0,0))
            continue
        
        points = []
        for point in contour:
            xa, ya = point[0]
            xb, yb = center
            distance = math.sqrt((xb - xa)**2 + (yb - ya)**2)
            points.append(distance)
            print(f'Distance: {distance}')

        if min(points) > min(closest_pnt):
            cv2.fillPoly(result, pts=[contour], color=(0,0,0))
            continue
    
    return result

image = cv2.imread('data/image_2892_4.png')
plt.imshow(preprocess(image))
plt.show()