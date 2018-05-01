import cv2 as cv
import numpy as np

APPROX_POLY_DP_ERROR = 0.05
MIN_SQUARE_AREA = 500


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img):
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
                if len(cnt) == 4 and cv.contourArea(cnt) > MIN_SQUARE_AREA and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def main():
    cap = cv.VideoCapture('./stackerVids/IMG_0565.MOV')

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Filter out noise based on color?
        lower_blue = np.array([192, 206, 68])  # Darker teal
        upper_blue = np.array([253, 255, 244])  # Lighter teal (almost white!)
        mask = cv.inRange(frame, lower_blue, upper_blue)
        output = cv.bitwise_and(frame, frame, mask=mask)

        # Morphology
        kernel = np.ones((10, 10), np.uint8)
        closing = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)

        # Square Detection
        squares = find_squares(closing)
        cv.drawContours(closing, squares, -1, (0, 255, 0), 3)

        # Resizing
        closing = cv.resize(closing, (0, 0), fx=0.5, fy=0.5)

        cv.imshow('frame', closing)

        if cv.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
