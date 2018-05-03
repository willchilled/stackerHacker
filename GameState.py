from enum import Enum
import numpy as np


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

    def check_game_begun(self, centres, average_square_width):
        if len(centres) < 3:
            return False
        centres = list(map(lambda x: np.array(x), centres))
        # Returns true if every centre is at least within 10% diff of the avg square width from another centre
        return all(any(abs(np.linalg.norm(centres[j] - centres[i])) < 0.1 * average_square_width for j in
                       range(i + 1, len(centres))) for i in range(len(centres) - 1))
