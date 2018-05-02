import cv2 as cv
import numpy as np
import time

from frameProcessor import FrameProcessor
from GameState import *
from utils import *

LOWER_BLUE = np.array([192, 206, 68])  # Darker teal
UPPER_BLUE = np.array([253, 255, 244])  # Lighter teal (almost white!)
SQUARE_SIZE = 2000


def check_game_begun(centres, average_square_width):
    centres = list(map(lambda x: np.array(x), centres))
    # Returns true if every centre is at least within 10% diff of the avg square width from another centre
    return all(any(
        abs(np.linalg.norm(centres[j] - centres[i])) < 0.1 * average_square_width for j in range(i + 1, len(centres)))
               for i in range(len(centres) - 1))


def init_game():
    pass


def find_top_tower_cols(frame):
    pass


class StackerController:

    def __init__(self, input_cam):
        self.cam = cv.VideoCapture(input_cam)
        self.game_state = GameState()
        self.fp = FrameProcessor(LOWER_BLUE, UPPER_BLUE, SQUARE_SIZE)

        self.has_init = False
        self.avg_sq_width = -1

        ret, self.frame_0 = self.cam.read()
        ret, self.frame_1 = self.cam.read()
        self.display_frame = self.frame_1

    def run(self):
        while self.cam.isOpened():

            if self.game_state.stage == Stage.PREGAME:
                # frame = self.fp.colour_segment(self.frame_1)
                frame = self.fp.frame_diff(self.frame_0, self.frame_1)
                self.display_frame, squares, centres = self.fp.detect_squares(frame)

                # If we haven't initialised and there are at least 5 contours to init off
                if not self.has_init and len(centres) >= 3:
                    avg_sq_width = get_average_box_width(squares)
                    if avg_sq_width != -1:
                        self.avg_sq_width = avg_sq_width
                        self.fp.square_size = avg_sq_width * avg_sq_width
                        # TODO add in working out columns
                        # TODO work out block colours dynamically
                        print("average_square_size", avg_sq_width * avg_sq_width)
                        self.has_init = True

                # If we have initialised now check if the game has begun
                elif self.has_init and check_game_begun(centres, self.avg_sq_width):
                    self.game_state.stage = Stage.PLAYING
                    print("We playing now!")

            elif self.game_state.stage == Stage.PLAYING:
                frame = self.fp.frame_diff(self.frame_0, self.frame_1)
                self.display_frame, squares, centres = self.fp.detect_squares(frame, True)
                tower_cols = find_top_tower_cols(self.display_frame)

            self.frame_1 = self.frame_0
            ret, self.frame_0 = self.cam.read()

            if not ret:
                break

            self.display_frame = cv.resize(self.display_frame, (0, 0), fx=0.5, fy=0.5)
            cv.imshow("frame", self.display_frame)

            if cv.waitKey(50) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv.destroyAllWindows()
