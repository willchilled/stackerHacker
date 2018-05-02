from enum import Enum


class Stage(Enum):
    PREGAME = 1
    PLAYING = 2


class GameState:

    def __init__(self):
        self.stage = Stage.PREGAME
        self.tower_cols = -1
        self.tower_height = 0
        self.moving_block_0 = -1
        self.moving_block_1 = -1

    def update_state(self, frame, squares, centres):
        pass
