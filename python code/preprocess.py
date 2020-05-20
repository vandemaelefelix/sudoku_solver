import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('data/image_2897_1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 3)
img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
result = cv2.bitwise_not(img, img)

contours, _ = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

largest_contour = 0
center_contours = []

border_tresh = 2

for contour in contours:
    for point in contour:
        if (abs(0 - point[0][0]) < border_tresh) or (abs(0 - point[0][1]) < border_tresh) or (abs(image.shape[0] - point[0][0]) < border_tresh) or (abs(image.shape[0] - point[0][1]) < border_tresh):
            cv2.fillPoly(result, pts=[contour], color=(0,0,0))
            break
    else:
        center_contours.append(contour)
        area = cv2.contourArea(contour)
        if area > largest_contour:
            largest_contour = area

print(largest_contour)

for contour in center_contours:
    area = cv2.contourArea(contour)
    if area < largest_contour or area < 30:
        cv2.fillPoly(result, pts=[contour], color=(0,0,0))
        pass

kernel = np.ones((3, 3), np.uint8) 
result = cv2.erode(result, kernel)
result = cv2.dilate(result, kernel)


plt.imshow(result)
plt.show()