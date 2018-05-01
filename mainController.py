from enum import Enum
import cv2 as cv
import numpy as np
import time

from frameProcessor import FrameProcessor

LOWER_BLUE = np.array([192, 206, 68])  # Darker teal
UPPER_BLUE = np.array([253, 255, 244])  # Lighter teal (almost white!)
SQUARE_SIZE = 500


class Stage(Enum):
    INIT = 1
    PLAYING = 2


def check_game_begun(frame):
    pass


def init_game():
    pass


def find_top_tower_cols(frame):
    pass


def main():
    fp = FrameProcessor(LOWER_BLUE, UPPER_BLUE, SQUARE_SIZE)
    cap = cv.VideoCapture('./stackerVids/IMG_0565.MOV')
    stage = Stage.INIT

    tower_cols = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, squares, centres = fp.process_frame(frame)

        if stage == Stage.INIT:
            if check_game_begun(frame):
                init_game()
                stage = Stage.PLAYING

        if stage == Stage.PLAYING:
            tower_cols = find_top_tower_cols(frame)

        frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv.imshow("frame", frame)

        if cv.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
