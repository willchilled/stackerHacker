import cv2 as cv
import numpy as np
import datetime
import time

from frameProcessor import *
from GameState import *
from utils import *

STACKER_COLUMNS = 7
STACKER_ROWS = 15
FPS = 30

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

        self.sample_time = time.time()

        # Velocity sampling times
        self.frame_diff = None
        self.last_frame_sample = None

        # Camera Variables
        self.cam = cv.VideoCapture(input_cam)
        ret, self.frame_0 = self.cam.read()
        ret, self.frame_1 = self.cam.read()
        self.display_frame = self.frame_1
        self.frame_count = 2

    def run(self):
        print_divider("STAGE=PREGAME")
        while self.cam.isOpened():

            # PREGAME
            if self.game_stage == Stage.PREGAME:
                self.pregame_task()

            # INIT
            elif self.game_stage == Stage.INIT:
                self.init_task()

            # PLAYING
            elif self.game_stage == Stage.PLAYING:
                self.play_task()

            # UPDATE FRAMES
            self.frame_1 = self.frame_0
            ret, self.frame_0 = self.cam.read()
            self.frame_count += 1
            if not ret:
                break

            # DRAW DISPLAY FRAME
            frame = cv.resize(self.display_frame, (0, 0), fx=0.5, fy=0.5)
            cv.imshow("The gamey boi", frame)
            if cv.waitKey(50) & 0xFF == ord('q'):
                break

        # TIDY UP ON CAMERA CLOSE
        self.cam.release()
        cv.destroyAllWindows()

    def pregame_task(self):
        """Handles:
                Calculating average size of squares to look for
                TODO: Dynamically calculate angle to rotate screen by
                TODO: Calculate block colours dynamically maybe
        """

        frame = frame_diff(self.frame_0, self.frame_1)
        self.display_frame = morph_open(frame)
        squares, centres = detect_squares(frame)

        # If we haven't initialised and there are at least 5 contours to init off
        if not self.avg_sq_width and len(centres) >= 5:
            avg_sq_width = get_average_box_width(squares)
            if avg_sq_width != -1:
                self.avg_sq_size = avg_sq_width * avg_sq_width
                self.avg_sq_width = avg_sq_width

                # TODO add in working out columns

                # TODO Work out rotation angle dynamically
                self.r_angle = -2

                # TODO work out block colours dynamically
                print("average_square_size", self.avg_sq_size)

        if self.avg_sq_width and len(centres) == 1:
            bounging_box = cv.boundingRect(squares[0])
            if not self.last_bb:
                self.last_sq, self.last_bb = squares[0], bounging_box
            else:
                # On the same row
                if abs(bounging_box[1] - self.last_bb[1]) < 0.1 * self.avg_sq_width:
                    # Moved one column
                    self.game_stage = Stage.INIT
                    print_divider("STAGE=INIT")
                    self.last_bb, self.last_sq = None, None

                self.last_sq, self.last_bb = squares[0], bounging_box

    def init_task(self):
        """Handles:
                Calculating the leftmost square possible
        """
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

    def play_task(self):
        """Handles:
                Detecting Layer increment
                Calculating period of a layer
        """
        # Detect squares
        frame = frame_diff(self.frame_0, self.frame_1)
        frame = morph_open(frame)
        frame = rotate_frame(frame, self.r_angle)
        squares, centres = detect_squares(frame, self.avg_sq_size, True)
        highest_cnt, highest_bb = get_highest_square(squares)

        if highest_bb:
            # If the new highest contour is one row above the last square. Increment current row
            if self.avg_sq_width * 0.5 < self.current_row_height - highest_bb[1] < self.avg_sq_width * 2:
                self.current_row_height = highest_bb[1]
                self.current_row += 1
                self.tower_squares += [self.last_sq]
                self.last_frame_sample = None
                self.frame_diff = None
                print("current row", self.current_row)

            # The new highest contour is on the same row as the last square
            elif abs(self.current_row_height - highest_bb[1]) < 0.1 * self.avg_sq_width:

                # The square has moved one block horizontally over since the last frame.
                if self.last_bb and self.avg_sq_width * 0.5 < abs(self.last_bb[0] - highest_bb[0]) < self.avg_sq_width * 1.5:
                    if not self.last_frame_sample:
                        self.last_frame_sample = self.frame_count

                    # We have the time interval between a block moving from one position to the next
                    elif not self.frame_diff:
                        self.frame_diff = self.frame_count - self.last_frame_sample
                        current_x = highest_bb[0]
                        if self.tower_squares:
                            target_x = get_highest_square(self.tower_squares)[1][0]
                            block_distance = int(abs(target_x - current_x) // (self.avg_sq_width * 1.2))
                        else:
                            target_x = "ANY"
                            block_distance = 0

                        print_divider("currentX {}\ntargetX {}\nblockDistance {}\nBlock Speed {:.1f}".format(current_x, target_x, block_distance, 1 / (self.frame_diff / FPS)))



                # Update the last highest square
                self.last_sq, self.last_bb = highest_cnt, highest_bb


        # Draw onto original image
        display_frame = rotate_frame(self.frame_0, self.r_angle)
        display_frame = cv.drawContours(display_frame, self.square_contours, -1, GREEN, 1)
        display_frame = cv.drawContours(display_frame, self.tower_squares, -1, ORANGE, 5)
        display_frame = cv.drawContours(display_frame, squares, -1, RED, 4)

        self.square_contours += squares
        self.display_frame = display_frame
