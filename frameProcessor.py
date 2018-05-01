import cv2 as cv
import numpy as np

APPROX_POLY_DP_ERROR = 0.05


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img, square_area):
    # TODO Work out how to stop contours matching over each other
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            bin, contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, APPROX_POLY_DP_ERROR * cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > square_area * 0.9 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


class FrameProcessor:

    def __init__(self, lower_blue, upper_blue, square_size):
        self.lower_blue = lower_blue
        self.upper_blue = upper_blue
        self.square_size = square_size

    def process_frame(self, frame):
        mask = cv.inRange(frame, self.lower_blue, self.upper_blue)
        output = cv.bitwise_and(frame, frame, mask=mask)

        # Morphology
        kernel = np.ones((10, 10), np.uint8)
        closing = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)

        # Square Detection
        squares = find_squares(closing, self.square_size)
        cv.drawContours(closing, squares, -1, (0, 255, 0), 3)

        # Compute the center of each square and draw
        centres = []
        for square in squares:
            M = cv.moments(square)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.circle(closing, (cX, cY), 5, (0, 0, 255), -1)
            centres.append((cX, cY))

        return closing, squares, centres
