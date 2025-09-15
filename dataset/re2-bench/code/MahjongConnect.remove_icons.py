
import random

class MahjongConnect():

    def __init__(self, BOARD_SIZE, ICONS):
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()

    def remove_icons(self, pos1, pos2):
        (x1, y1) = pos1
        (x2, y2) = pos2
        self.board[x1][y1] = ' '
        self.board[x2][y2] = ' '
