import cv2 as cv
import numpy as np

DIVIDER_SIZE = 20


def get_average_box_width(contours):
    total_box_widths = 0
    num_box_widths = 0
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if (h * 0.9) < w < (h * 1.1):
            total_box_widths += w
            num_box_widths += 1

    if num_box_widths == 0:
        return -1

    return total_box_widths / num_box_widths


def print_divider(message):
    print("-" * DIVIDER_SIZE)
    print(message)
    print("-" * DIVIDER_SIZE)


def get_highest_square(squares):
    cnt, bounding_box = None, None
    if squares:
        boundingBoxes = [cv.boundingRect(c) for c in squares]
        cnt, bounding_box = min(zip(squares, boundingBoxes), key=lambda x: x[1][1])
    return cnt, bounding_box


def get_leftmost_square(squares):
    cnt, bounding_box = None, None
    if squares:
        boundingBoxes = [cv.boundingRect(c) for c in squares]
        cnt, bounding_box = min(zip(squares, boundingBoxes), key=lambda x: x[1][0])
    return cnt, bounding_box


def get_avg_contour_color(frame, contour):
    # TODO may have to convert frame to gray first
    mask = np.zeros(frame.shape, np.uint8)
    cv.drawContours(mask, [contour], 0, 255, -1)
    mean_val = cv.mean(frame, mask=mask)

    print("mean_val: ", mean_val)