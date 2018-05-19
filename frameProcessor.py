import cv2 as cv
import numpy as np

APPROX_POLY_DP_ERROR = 0.05
BLOCK_TO_SCREEN_FACTOR = 70
KERNEL = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
ROTATE_ANGLE = -2
SQUARE_SIZE = 2000

LOWER_BLUE = np.array([192, 206, 68])  # Darker teal
UPPER_BLUE = np.array([253, 255, 244])  # Lighter teal (almost white!)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
ORANGE = (0, 102, 255)
YELLOW = (0, 255, 255)


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

    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        # for thrs in range(0, 255, 26):
        thrs = 10
        _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
        if tight_match:
            bin, contours, _hierarchy = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        else:
            bin, contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cnt_len = cv.arcLength(cnt, True)

            # Approximate all contours in the image to reduces set of points of themselves
            cnt = cv.approxPolyDP(cnt, APPROX_POLY_DP_ERROR * cnt_len, True)

            # Square check - Four points to the contour, convex,
            if is_square(cnt, square_area, tight_match):
                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                if max_cos < 0.1:
                    squares.append(cnt)
    return squares


# Not used any longer (It sucked)
def colour_segment(frame):
    mask = cv.inRange(frame, LOWER_BLUE, UPPER_BLUE)
    output = cv.bitwise_and(frame, frame, mask=mask)

    return output


def detect_squares(frame, square_size=SQUARE_SIZE, tight_match=False):
    squares = find_squares(frame, square_size, tight_match)

    # Compute the center of each square and draw
    centres = []
    for square in squares:
        M = cv.moments(square)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centres.append((cX, cY))

    return squares, centres


def frame_diff(frame1, frame2):
    img1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    diff = cv.subtract(img1, img2)

    return diff


def rotate_frame(frame, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = frame.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(frame, M, (nW, nH))


def morph_open(frame):
    return cv.morphologyEx(frame, cv.MORPH_OPEN, KERNEL)
