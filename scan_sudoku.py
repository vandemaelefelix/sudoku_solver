import cv2
import math
import numpy as np
import operator
import os
import glob

# from PIL import Image  
# import PIL

# image = cv2.imread('images\IMG_20200501_150800.jpg')
# image = cv2.resize(image, (720, 720))
# image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# print(image.shape)

def preprocess_image(image):
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed_img = cv2.GaussianBlur(processed_img.copy(), (9, 9), 3)

    # Binary adaptive threshold using 11 nearest neighbour pixels
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
    # Locate points of the documents or object which you want to transform 
    pts1 = np.float32(pts)
    # pts1 = np.float32([[76, 91], [559,  81], [624, 601], [  0, 599]]) 
    pts2 = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]]) 
        
    # Apply Perspective Transform Algorithm 
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    result = cv2.warpPerspective(image, matrix, (640, 640))

    return result

def loop_boxes(img):
    sq_width = int(img.shape[0] / 9)
    sq_height = int(img.shape[1] / 9)

    boxes = []
    for i in range(9):
        for j in range(9):
            crop_img = img[i*sq_height:i*sq_height+sq_height, j*sq_width:j*sq_width+sq_width]
            boxes.append(crop_img)

    return boxes

def get_name_count(path):
    import glob
    import os

    list_of_files = glob.glob(f'{path}/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    
    return latest_file.split('_')[1]

def display_images(folder_path):
    img_dir = folder_path # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.resize(img, (720, 720))
        data.append(img)

    for i in data:
        cv2.imshow('image', i)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows
            break

def get_images(folder_path):
    img_dir = folder_path # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    images = []
    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.resize(img, (720, 720))
        images.append(img)

    return images

def create_dataset(images, skip_zero=False, count=0):
    print(len(images))
    while 1:
        cv2.imshow('image', images[count])
        key = cv2.waitKey(0)
        print(key)
        if key == 27:
            cv2.destroyAllWindows 
            break
        elif key == 0:
            images[count] = cv2.rotate(images[count], cv2.ROTATE_90_COUNTERCLOCKWISE)
            create_dataset(images, count)
            break
        elif key == 13:
            image = images[count]
            corners = find_corners(preprocess_image(image))
            result = four_point_transform(corners, image)

            number_imgs = loop_boxes(result)

            count_number = int(get_name_count('data/'))

            for box in number_imgs:
                cv2.imshow('number', box)

                key = cv2.waitKey(0)
                print(key)
                if key == 27:
                    cv2.destroyAllWindows()
                    break

                if key == 46:
                    continue                
                
                if key >= 0:
                    number = key - 48
                    if skip_zero and number == 0:
                        continue
                    count_number += 1
                    print(f'data/{count_number}/{number}.png')
                    cv2.imwrite(f'data/image_{count_number}_{number}.png', box)
            cv2.destroyWindow('number')

        if count != len(images) - 1:
            count += 1
        else: 
            count = 0

create_dataset(get_images('images_v2/'))

# display_images('images/')