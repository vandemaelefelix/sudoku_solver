import cv2
import math
import numpy as np
import operator
import os
import glob
import shutil

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

def get_number_count(path):
    with open(path, "r") as file:
        count = file.readline()
        file.close()
    return count

def write_number_count(path, count):
    with open(path, 'w') as file:
        file.write(count)
        file.close()

def create_dataset(folder_path, images_folder_path, skip_zero=False, count=0):
    img_dir = folder_path
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    count_number = int(get_number_count('count.txt'))

    for f1 in files:
        print(f1)
        img = cv2.imread(f1)
        img = cv2.resize(img, (720, 720))

        cv2.namedWindow(f1)
        cv2.moveWindow(f1, 500,100)

        cv2.imshow(f1, img)
        key = cv2.waitKey(0)
        if key == 27: # ==> escape
            cv2.destroyAllWindows()
            break
        elif key == 32: # ==> spacebar
            cv2.destroyWindow(f1)
            continue
        elif key == 13: # ==> enter
            corners = find_corners(preprocess_image(img))
            result = four_point_transform(corners, img)

            number_imgs = loop_boxes(result)

            for box in number_imgs:
                num_img = box
                cv2.namedWindow('number')
                cv2.moveWindow('number', 1220,100)
                cv2.imshow('number', cv2.resize(num_img, (200, 200)))
                key = cv2.waitKey(0)
                if key == 27: # ==> escape
                    cv2.destroyWindow('number')
                    break
                if 48 <= key <= 57:
                    number = key - 48
                    count_number += 1
                    print(f'data/{count_number}/{number}.png')
                    cv2.imwrite(f'data/image_{count_number}_{number}.png', box)
    
                cv2.destroyWindow('number')

            new_path = "{}{}".format(images_folder_path, f1.split("\\")[1])
            shutil.move(f1, new_path)
            
        cv2.destroyWindow(f1)

    write_number_count('count.txt', str(count_number))





create_dataset('images_v2/', 'images/')
