

class GomokuGame():

    def __init__(self, board_size):
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 'X'

    def make_move(self, row, col):
        if (self.board[row][col] == ' '):
            self.board[row][col] = self.current_player
            self.current_player = ('O' if (self.current_player == 'X') else 'X')
            return True
        return False
