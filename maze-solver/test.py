#!/usr/bin/env python3

import os
import cv2
import numpy as np

if __name__ == '__main__':
    image_file = 'maze.png'
    if not os.path.exists(image_file):
        print('\n[Error] Image not found')
        exit()
    original_img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    original_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    height, widht = original_img[:, :, 0].shape
    original_img = cv2.resize(original_img, (height, height))
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(img, (3, 3))

    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=1,
                               maxRadius=40)

    print(circles)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for pt in circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(original_img, (a, b), r + 5, (0, 255, 0), 2)

    print(circles)

    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Original Image', original_img)
    cv2.waitKey(0)
    cv2.imshow('Thresholded Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
