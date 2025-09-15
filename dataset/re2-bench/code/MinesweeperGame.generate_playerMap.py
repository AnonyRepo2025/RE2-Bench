
import random

class MinesweeperGame():

    def __init__(self, n, k) -> None:
        self.n = n
        self.k = k
        self.minesweeper_map = self.generate_mine_sweeper_map()
        self.player_map = self.generate_playerMap()
        self.score = 0

    def generate_playerMap(self):
        arr = [['-' for row in range(self.n)] for column in range(self.n)]
        return arr
