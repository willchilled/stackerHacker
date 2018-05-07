import cv2 as cv
import numpy as np
import time

from frameProcessor import FrameProcessor
from GameState import *
from utils import *

LOWER_BLUE = np.array([192, 206, 68])  # Darker teal
UPPER_BLUE = np.array([253, 255, 244])  # Lighter teal (almost white!)
SQUARE_SIZE = 2000
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


def get_highest_square(squares):
    cnt = None
    if squares:
        boundingBoxes = [cv.boundingRect(c) for c in squares]
        cnt, _ = min(zip(squares, boundingBoxes), key=lambda x: x[1][1])
    return cnt


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

        self.square_contours = []

    def run(self):
        while self.cam.isOpened():

            if self.game_state.stage == Stage.PREGAME:
                frame = self.fp.frame_diff(self.frame_0, self.frame_1)
                self.display_frame = self.fp.morph_open(frame)
                squares, centres = self.fp.detect_squares(frame)

                # If we haven't initialised and there are at least 5 contours to init off
                if not self.has_init and len(centres) >= 3:
                    avg_sq_width = get_average_box_width(squares)
                    if avg_sq_width != -1:
                        self.avg_sq_width = avg_sq_width
                        self.fp.square_size = avg_sq_width * avg_sq_width

                        # TODO add in working out columns

                        # TODO Work out rotation angle
                        self.fp.r_angle = -1.5

                        # TODO work out block colours dynamically
                        print("average_square_size", avg_sq_width * avg_sq_width)
                        self.has_init = True

                # If we have initialised now check if the game has begun
                elif self.has_init and self.game_state.check_game_begun(centres, self.avg_sq_width):
                    self.game_state.stage = Stage.PLAYING
                    print("We playing now!")

            elif self.game_state.stage == Stage.PLAYING:

                # Detect squares
                frame = self.fp.frame_diff(self.frame_0, self.frame_1)
                frame = self.fp.morph_open(frame)
                frame = self.fp.rotate_frame(frame)
                squares, centres = self.fp.detect_squares(frame, True)
                highest_cnt = get_highest_square(self.square_contours)

                # Draw onto original image
                display_frame = self.fp.rotate_frame(self.frame_1)
                display_frame = cv.drawContours(display_frame, self.square_contours, -1, BLUE, 3)
                display_frame = cv.drawContours(display_frame, squares, -1, RED, 4)
                display_frame = cv.drawContours(display_frame, [highest_cnt], -1, GREEN, 5)
                self.square_contours += squares

                self.display_frame = display_frame

            self.frame_1 = self.frame_0
            ret, self.frame_0 = self.cam.read()

            if not ret:
                break

            frame = cv.resize(self.display_frame, (0, 0), fx=0.5, fy=0.5)
            cv.imshow("frame", frame)

            if cv.waitKey(50) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv.destroyAllWindows()
