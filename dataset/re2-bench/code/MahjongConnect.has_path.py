
import random

class MahjongConnect():

    def __init__(self, BOARD_SIZE, ICONS):
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()

    def has_path(self, pos1, pos2):
        visited = set()
        stack = [pos1]
        while stack:
            current_pos = stack.pop()
            if (current_pos == pos2):
                return True
            if (current_pos in visited):
                continue
            visited.add(current_pos)
            (x, y) = current_pos
            for (dx, dy) in [(0, 1), (0, (- 1)), (1, 0), ((- 1), 0)]:
                (new_x, new_y) = ((x + dx), (y + dy))
                if ((0 <= new_x < self.BOARD_SIZE[0]) and (0 <= new_y < self.BOARD_SIZE[1])):
                    if (((new_x, new_y) not in visited) and (self.board[new_x][new_y] == self.board[x][y])):
                        stack.append((new_x, new_y))
        return False
