import cv2 as cv


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