

class PushBoxGame():

    def __init__(self, map):
        self.map = map
        self.player_row = 0
        self.player_col = 0
        self.targets = []
        self.boxes = []
        self.target_count = 0
        self.is_game_over = False
        self.init_game()

    def init_game(self):
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                if (self.map[row][col] == 'O'):
                    self.player_row = row
                    self.player_col = col
                elif (self.map[row][col] == 'G'):
                    self.targets.append((row, col))
                    self.target_count += 1
                elif (self.map[row][col] == 'X'):
                    self.boxes.append((row, col))
