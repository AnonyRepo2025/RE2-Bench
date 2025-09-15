

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

    def check_win(self):
        box_on_target_count = 0
        for box in self.boxes:
            if (box in self.targets):
                box_on_target_count += 1
        if (box_on_target_count == self.target_count):
            self.is_game_over = True
        return self.is_game_over
