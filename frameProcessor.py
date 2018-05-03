import cv2 as cv
import numpy as np

APPROX_POLY_DP_ERROR = 0.05
BLOCK_TO_SCREEN_FACTOR = 70


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def is_square(cnt, square_area=2000, tight_match=False):
    if tight_match:
        return len(cnt) == 4 and abs(cv.contourArea(cnt) - square_area) < square_area * 0.1 and cv.isContourConvex(cnt)
    else:
        return len(cnt) == 4 and cv.contourArea(cnt) > square_area * 0.9 and cv.isContourConvex(cnt)


def find_squares(img, square_area=2000, tight_match=False):
    # TODO Work out how to stop contours matching over each other

    height, width = img.shape[:2]
    blank_image = np.zeros((width, height), np.uint8)

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

                # Check if the contour is overlapping with another, if so do not continue processing
                if any(blank_image[i, j] for i, j in cnt[0]):
                    continue
                else:
                    for i, j in cnt[0]:
                        blank_image[i, j] = 1

                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, APPROX_POLY_DP_ERROR * cnt_len, True)
                if is_square(cnt, square_area, tight_match):
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

    def colour_segment(self, frame):
        mask = cv.inRange(frame, self.lower_blue, self.upper_blue)
        output = cv.bitwise_and(frame, frame, mask=mask)

        return output

    def detect_squares(self, frame, tight_match=False):
        squares = find_squares(frame, self.square_size, tight_match)
        cv.drawContours(frame, squares, -1, (255, 0, 0), 3)

        # Compute the center of each square and draw
        centres = []
        for square in squares:
            M = cv.moments(square)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            centres.append((cX, cY))

        return frame, squares, centres

    def frame_diff(self, frame1, frame2):
        img1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        diff = cv.subtract(img1, img2)

        return diff

