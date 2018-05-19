from enum import Enum
import numpy as np


class Stage(Enum):
    PREGAME = 1
    INIT = 2
    PLAYING = 3


class GameState():

    def __init__(self):
        self.stage = Stage.PREGAME
        self.is_moving_left = None
        self.left_most_sq = None
        self.left_most_bb = None
        self.screen_boundin_b = None


def update_state(frame, squares, centres):
    pass
