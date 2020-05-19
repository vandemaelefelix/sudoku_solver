import cv2
import numpy as np
import operator
import copy
import matplotlib.pyplot as plt
import os, glob

def preprocess_image(image):
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed_img = cv2.GaussianBlur(processed_img.copy(), (9, 9), 3)

    processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    processed_img = cv2.bitwise_not(processed_img, processed_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    processed_img = cv2.dilate(processed_img, kernel)

    return processed_img

def find_corners(image):
    ext_contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))

    return [contour[top_left][0], contour[top_right][0], contour[bottom_right][0], contour[bottom_left][0]]

def four_point_transform(pts, image):
    pts1 = np.float32(pts)
    pts2 = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]]) 
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    result = cv2.warpPerspective(image, matrix, (640, 640))

    return result


if __name__ == '__main__':
    # filname = os.path.join('images_v3/image162.jpg')

    img_dir = 'images_v3/'
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)

    for f1 in files:
        print(f1)
        image = cv2.imread(f1)
        shape = min(image.shape[0:1])

        preprocessed_image = preprocess_image(image)

        contours, _ = cv2.findContours(preprocessed_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        x,y,w,h = cv2.boundingRect(contour)
        # result = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

        image = image[y:y+h, x:x+w]

        cv2.imwrite(f1, image)

    # plt.imshow(image)
    # plt.show()