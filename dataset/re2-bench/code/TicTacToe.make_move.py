

class TicTacToe():

    def __init__(self, N=3):
        self.board = [[' ' for _ in range(N)] for _ in range(3)]
        self.current_player = 'X'

    def make_move(self, row, col):
        if (self.board[row][col] == ' '):
            self.board[row][col] = self.current_player
            self.current_player = ('O' if (self.current_player == 'X') else 'X')
            return True
        else:
            return False
