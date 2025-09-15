
import random

class MinesweeperGame():

    def __init__(self, n, k) -> None:
        self.n = n
        self.k = k
        self.minesweeper_map = self.generate_mine_sweeper_map()
        self.player_map = self.generate_playerMap()
        self.score = 0

    def check_won(self, map):
        for i in range(self.n):
            for j in range(self.n):
                if ((map[i][j] == '-') and (self.minesweeper_map[i][j] != 'X')):
                    return False
        return True
