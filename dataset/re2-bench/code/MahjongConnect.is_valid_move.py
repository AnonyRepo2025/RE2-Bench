
import random

class MahjongConnect():

    def __init__(self, BOARD_SIZE, ICONS):
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()

    def is_valid_move(self, pos1, pos2):
        (x1, y1) = pos1
        (x2, y2) = pos2
        if (not ((0 <= x1 < self.BOARD_SIZE[0]) and (0 <= y1 < self.BOARD_SIZE[1]) and (0 <= x2 < self.BOARD_SIZE[0]) and (0 <= y2 < self.BOARD_SIZE[1]))):
            return False
        if (pos1 == pos2):
            return False
        if (self.board[x1][y1] != self.board[x2][y2]):
            return False
        if (not self.has_path(pos1, pos2)):
            return False
        return True

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
    
class Test(unittest.TestCase):
    def test(self):
            mc = MahjongConnect([4, 4], ['a', 'b', 'c'])
            mc.board = [['a', 'b', 'c', 'a'],
                        ['a', 'b', 'c', 'a'],
                        ['a', 'b', 'c', 'a'],
                        ['a', 'b', 'c', 'a']]
            res = mc.is_valid_move((0, 0), (1, 0))
            return res