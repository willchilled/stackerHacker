from enum import Enum
import cv2 as cv
import numpy as np
import time

from frameProcessor import FrameProcessor
from utils import *

LOWER_BLUE = np.array([192, 206, 68])  # Darker teal
UPPER_BLUE = np.array([253, 255, 244])  # Lighter teal (almost white!)
SQUARE_SIZE = 500


class Stage(Enum):
    PREGAME = 1
    PLAYING = 2


def check_game_begun(centres, average_square_width):
    print("checking for game beginning... len of centres=", len(centres))
    # if len(centres) != 3:
    #     return False

    centres = list(map(lambda x: np.array(x), centres))
    # hehehehe
    return all(any(abs(np.linalg.norm(centres[j]-centres[i])) < 0.1 * average_square_width for j in range(i+1, len(centres)))for i in range(len(centres) - 1))





def init_game():
    pass


def find_top_tower_cols(frame):
    pass


def main():
    fp = FrameProcessor(LOWER_BLUE, UPPER_BLUE, SQUARE_SIZE)
    cap = cv.VideoCapture('./stackerVids/IMG_0565.MOV')
    stage = Stage.PREGAME
    has_init = False
    average_square_width = -1

    tower_cols = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, squares, centres = fp.process_frame(frame)

        if stage == Stage.PREGAME:
            # If we haven't initialised and there are at least 5 contours to init off
            if not has_init and len(centres) > 5:
                average_square_width = get_average_box_width(squares)
                if average_square_width != -1:
                    fp.square_size = average_square_width * average_square_width
                    # TODO add in working out columns
                    # TODO work out block colours dynamically
                    print("average_square_size", average_square_width * average_square_width)
                    has_init = True

            # If we have initialised now check if the game has begun
            elif has_init and check_game_begun(centres, average_square_width):
                stage = Stage.PLAYING

        if stage == Stage.PLAYING:
            print("now we playing home slice")
            tower_cols = find_top_tower_cols(frame)

        frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv.imshow("frame", frame)

        if cv.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
