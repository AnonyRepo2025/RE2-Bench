
import random

class MahjongConnect():

    def __init__(self, BOARD_SIZE, ICONS):
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()

    def is_game_over(self):
        for row in self.board:
            if any(((icon != ' ') for icon in row)):
                return False
        return True
