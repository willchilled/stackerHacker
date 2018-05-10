import cv2 as cv
import numpy as np
import time

from frameProcessor import *
from GameState import *
from utils import *
from scipy import ndimage

STACKER_COLUMNS = 7
STACKER_ROWS = 15

DIVIDER_SIZE = 20

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


class StackerController:

    def __init__(self, input_cam):
        self.game_stage = Stage.PREGAME

        # Pregame init Variables
        self.avg_sq_size = None
        self.avg_sq_width = None
        self.r_angle = -1

        # Init variables
        self.is_moving_left = None
        self.left_most_sq = None
        self.left_most_bb = None

        # Playing Variables
        self.current_row, self.current_row_height = 0, None
        self.square_contours, self.tower_squares = [], []
        self.last_sq, self.last_bb = None, None

        # Camera Variables
        self.cam = cv.VideoCapture(input_cam)
        ret, self.frame_0 = self.cam.read()
        ret, self.frame_1 = self.cam.read()
        self.display_frame = self.frame_1

    def run(self):
        print_divider("STAGE=PREGAME")
        while self.cam.isOpened():

            # PREGAME
            if self.game_stage == Stage.PREGAME:
                frame = frame_diff(self.frame_0, self.frame_1)
                self.display_frame = morph_open(frame)
                squares, centres = detect_squares(frame)

                # If we haven't initialised and there are at least 5 contours to init off
                if not self.avg_sq_width and len(centres) >= 3:
                    avg_sq_width = get_average_box_width(squares)
                    if avg_sq_width != -1:
                        self.avg_sq_size = avg_sq_width * avg_sq_width
                        self.avg_sq_width = avg_sq_width

                        # TODO add in working out columns

                        # TODO Work out rotation angle
                        self.r_angle = -2

                        # TODO work out block colours dynamically
                        print("average_square_size", self.avg_sq_size)

                # If we have initialised now check if the game has begun
                elif self.avg_sq_width and check_game_begun(centres, self.avg_sq_width):
                    self.game_stage = Stage.INIT
                    print_divider("STAGE=INIT")

            # INIT
            elif self.game_stage == Stage.INIT:
                frame = frame_diff(self.frame_0, self.frame_1)
                frame = morph_open(frame)
                frame = rotate_frame(frame, self.r_angle)
                squares, centres = detect_squares(frame, self.avg_sq_size, True)
                square, bounding_b = get_leftmost_square(squares)

                if bounding_b:
                    if not self.left_most_bb:
                        self.left_most_bb = bounding_b
                    else:
                        if bounding_b[0] - self.left_most_bb[0] > self.avg_sq_width / 2:  # Moving Right
                            if self.is_moving_left:
                                print("Found leftmost Square!: ", self.left_most_bb)
                                print_divider("STAGE=PLAYING")
                                self.current_row_height = self.left_most_bb[1]
                                self.game_stage = Stage.PLAYING
                            self.is_moving_left = False
                        elif self.left_most_bb[0] - bounding_b[0] > self.avg_sq_width / 2:  # Moving Left
                            self.is_moving_left = True
                            self.left_most_bb = bounding_b
                            self.left_most_sq = square

                self.display_frame = rotate_frame(self.frame_1, self.r_angle)

            # PLAYING
            elif self.game_stage == Stage.PLAYING:

                # Detect squares
                frame = frame_diff(self.frame_0, self.frame_1)
                frame = morph_open(frame)
                frame = rotate_frame(frame, self.r_angle)
                squares, centres = detect_squares(frame, self.avg_sq_size, True)
                highest_cnt, bounding_b = get_highest_square(squares)

                # If the new highest contour is one row above increment it
                if bounding_b:
                    if self.current_row_height - bounding_b[1] > self.avg_sq_width / 2:
                        self.current_row_height = bounding_b[1]
                        self.current_row += 1
                        self.tower_squares += [self.last_sq]
                        print("current row, current_height", self.current_row, self.current_row_height)

                    if abs(self.current_row_height - bounding_b[1]) < 0.1 * self.avg_sq_width:
                        self.last_sq, self.last_bb = highest_cnt, bounding_b




                # Work some shit out
                if len(self.square_contours) > 3:
                    last_3 = self.square_contours[-3:]
                    # TODO do shit here next!

                # Draw onto original image
                display_frame = rotate_frame(self.frame_0, self.r_angle)
                display_frame = cv.drawContours(display_frame, self.square_contours, -1, GREEN, 1)
                display_frame = cv.drawContours(display_frame, self.tower_squares, -1, ORANGE, 5)
                display_frame = cv.drawContours(display_frame, squares, -1, RED, 4)


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
