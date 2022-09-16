#!/usr/bin/env python3

import os
import cv2
import numpy as np
from argparse import ArgumentParser


CELL_SIZE = 60

def read_image(img_file_path):
    if not os.path.exists(img_file_path):
        return None
    original_img = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
    original_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    height, width = original_img[:, :, 0].shape
    original_img = cv2.resize(original_img, (height, height))
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img, original_img

def find_start_and_finish(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(img, (3, 3))
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=1,
                               maxRadius=40)
    if circles is not None:
        circles = np.uint16(np.around(circles))
    return circles

def blockify(img, coord):
    size = CELL_SIZE
    h = size * (coord[0] + 1) # height of a cell
    w = size * (coord[1] + 1) # width of a cell
    h0 = size * coord[0]
    w0 = size * coord[1]
    block = img[h0:h, w0:w]

    up = bool(block[0, (size // 2)]) * 1000
    down = bool(block[(size - 1), (size // 2)]) * 100
    left = bool(block[(size // 2), 0]) * 10
    right = bool(block[(size // 2), (size - 1)]) * 1

    edge = up + down + left + right
    # print(block)
    return block, edge


def solve_maze(bin_img, start, stop, no_cells_height, no_cells_width):
    edge_array = []

    for i in range(no_cells_height):
        edge_array.append([])
        for j in range(no_cells_width):
            coord = [i, j]
            block, edge = blockify(bin_img, coord)
            edge_array[i].append(edge)

    edge = edge_array
   
    print(len(edge))

    # Using backtracking algorithm

    shortest_path = []
    sp = []
    rec = [0]
    img = bin_img
    ref = 0;
    p = 0
    sp.append(list(start))

    while True:
        h, w = sp[p][0], sp[p][1]
        print((h, w))
        if sp[-1] == list(stop):
            break

        if edge[h][w] > 0:
            rec.append(len(sp))

        if edge[h][w] > 999: # up
            edge[h][w] = edge[h][w] - 1000
            h = h - 1
            sp.append([h, w])
            edge[h][w] = edge[h][w] - 100
            p = p + 1
            continue

        if edge[h][w] > 99:
            edge[h][w] = edge[h][w] - 100
            h = h + 1
            sp.append([h,w])
            print((h, w))
            edge[h][w] = edge[h][w] - 1000
            p = p + 1
            continue

        if edge[h][w] > 9:
            edge[h][w] = edge[h][w] - 10
            w = w - 1
            sp.append([h,w])
            edge[h][w] = edge[h][w] - 1
            p = p + 1
            continue

        if edge[h][w] == 1:
            edge[h][w] = edge[h][w] - 1
            w = w + 1
            sp.append([h,w])
            edge[h][w] = edge[h][w] - 10
            p = p + 1
            continue

        else:
            sp.pop()
            rec.pop()
            p = rec[-1]
			
			
    for i in sp:
        shortest_path.append(tuple(i))

    return shortest_path


def color_cell(img, row, column, color_val):
    cell = img[CELL_SIZE * row:CELL_SIZE * (row + 1), CELL_SIZE * column:CELL_SIZE * (column + 1)]
    h, w = cell.shape
    if cell[int(h / 2)][int(w / 2)] != color_val:
        for row in range(h):
            for col in range(w):
                if cell[row][col] > color_val:
                    cell[row][col] = color_val

    return img


def highlight_path(img, start, stop, path):
    highlighted_img = img.copy()
    highlighted_img = color_cell(highlighted_img, start[0], start[1], 100)
    highlighted_img = color_cell(highlighted_img, stop[0], stop[1], 100)
    for i in path:
        highlighted_img = color_cell(highlighted_img, i[0], i[1], 200)

    return highlighted_img


def main():
    ap = ArgumentParser()
    ap.add_argument('-i', '--image', type=str, required=True,
                    help='Path to image file')
    args = vars(ap.parse_args())

    try:
        img, original_img = read_image(args['image'])
        height, width = img.shape
    except Exception as e:
        print('\n[Error] Error reading image')
        exit()

    print(height)
    print(width)

    circles = find_start_and_finish(original_img)

    for pt in circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        cv2.circle(original_img, (a, b), r + 5, (0, 255, 0), 2)
    
    no_cells_height = height // CELL_SIZE
    no_cells_width = width // CELL_SIZE
    print(no_cells_height)
    start = (0, 0)
    stop = ((no_cells_height - 1), (no_cells_width - 1))
    solved_img = None

    try:
        sp = solve_maze(img, start, stop, no_cells_height, no_cells_width)

        if len(sp) > 2:
            solved_img = highlight_path(img.copy(), start, stop, sp)
            print(sp)

        else:
            exit()

    except Exception as e:
        print('\n[ERROR] Failed to find shortest path')
        print(e.message)
        exit()

    cv2.imshow('Original Maze', original_img)
    cv2.waitKey(0)
    cv2.imshow('Solved Maze', solved_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(img[0:20, 0:20])

if __name__ == '__main__':
    main()
