import inspect
import json
import os
from datetime import datetime

def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def recursive_object_seralizer(obj, visited):
    seralized_dict = {}
    keys = list(obj.__dict__)
    for k in keys:
        if id(obj.__dict__[k]) in visited:
            seralized_dict[k] = "<RECURSIVE {}>".format(obj.__dict__[k])
            continue
        if isinstance(obj.__dict__[k], (float, int, str, bool, type(None))):
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], tuple):
            ## handle tuple
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], set):
            ## handle set
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], list):
            ## handle list
            seralized_dict[k] = obj.__dict__[k]
        elif hasattr(obj.__dict__[k], '__dict__'):
            ## handle object
            visited.append(id(obj.__dict__[k]))
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], dict):
            visited.append(id(obj.__dict__[k]))
            seralized_dict[k] = obj.__dict__[k]
        elif callable(obj.__dict__[k]):
            ## handle function
            if hasattr(obj.__dict__[k], '__name__'):
                seralized_dict[k] = "<function {}>".format(obj.__dict__[k].__name__)
        else:
            seralized_dict[k] = str(obj.__dict__[k])
    return seralized_dict

def inspect_code(func):
   def wrapper(*args, **kwargs):
       visited = []
       json_base = "/home/changshu/ClassEval/data/benchmark_solution_code/input-output/"
       if not os.path.exists(json_base):
           os.mkdir(json_base)
       jsonl_path = json_base + "/TicTacToe.jsonl"
       para_dict = {"name": func.__name__}
       args_names = inspect.getfullargspec(func).args
       if len(args) > 0 and hasattr(args[0], '__dict__') and args_names[0] == 'self':
           ## 'self'
           self_args = args[0]
           para_dict['self'] = recursive_object_seralizer(self_args, [id(self_args)])
       else:
           para_dict['self'] = {}
       if len(args) > 0 :
           if args_names[0] == 'self':
               other_args = {}
               for m,n in zip(args_names[1:], args[1:]):
                   other_args[m] = n
           else:
               other_args = {}
               for m,n in zip(args_names, args):
                   other_args[m] = n
           
           para_dict['args'] = other_args
       else:
           para_dict['args'] = {}
       if kwargs:
           para_dict['kwargs'] = kwargs
       else:
           para_dict['kwargs'] = {}
          
       result = func(*args, **kwargs)
       para_dict["return"] = result
       with open(jsonl_path, 'a') as f:
           f.write(json.dumps(para_dict, default=custom_serializer) + "\n")
       return result
   return wrapper


'''
# The class represents a game of Tic-Tac-Toe and its functions include making a move on the board, checking for a winner, and determining if the board is full.

class TicTacToe:
    def __init__(self, N=3):
        """
        Initialize a 3x3 game board with all empty spaces and current symble player, default is 'X'.
        """
        self.board = [[' ' for _ in range(N)] for _ in range(3)]
        self.current_player = 'X'

    def make_move(self, row, col):
        """
        Place the current player's mark at the specified position on the board and switch the mark.
        :param row: int, the row index of the position
        :param col: int, the column index of the position
        :return: bool, indicating whether the move was successful or not
        >>> ttt.current_player
        'X'
        >>> ttt.make_move(1, 1)
        >>> ttt.current_player
        'O'
        """

    def check_winner(self):
        """
        Check if there is a winner on the board in rows, columns and diagonals three directions
        :return: str or None, the mark of the winner ('X' or 'O'), or None if there is no winner yet
        >>> moves = [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2)]
        >>> for move in moves:
        ...     ttt.make_move(move[0], move[1])
        >>> ttt.check_winner()
        'X'
        """

    def is_board_full(self):
        """
        Check if the game board is completely filled.
        :return: bool, indicating whether the game board is full or not
        >>> ttt.is_board_full()
        False
        """
'''


class TicTacToe:
    def __init__(self, N=3):
        self.board = [[' ' for _ in range(N)] for _ in range(3)]
        self.current_player = 'X'

    @inspect_code
    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        else:
            return False

    @inspect_code
    def check_winner(self):
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        return None

    @inspect_code
    def is_board_full(self):
        for row in self.board:
            if ' ' in row:
                return False
        return True


import unittest

class TicTacToeTestMakeMove(unittest.TestCase):
    def test_make_move_1(self):
        ttt = TicTacToe()
        self.assertEqual(ttt.current_player, 'X')
        self.assertTrue(ttt.make_move(0, 0))
        self.assertEqual(ttt.current_player, 'O')

    # move invalid
    def test_make_move_2(self):
        ttt = TicTacToe()
        self.assertEqual(ttt.current_player, 'X')
        self.assertTrue(ttt.make_move(0, 0))
        self.assertTrue(ttt.make_move(0, 1))
        self.assertFalse(ttt.make_move(0, 0))
        self.assertEqual(ttt.current_player, 'X')

    def test_make_move_3(self):
        ttt = TicTacToe()
        self.assertEqual(ttt.current_player, 'X')
        self.assertTrue(ttt.make_move(0, 0))
        self.assertTrue(ttt.make_move(0, 1))
        self.assertTrue(ttt.make_move(1, 1))
        self.assertEqual(ttt.current_player, 'O')

    def test_make_move_4(self):
        ttt = TicTacToe()
        self.assertEqual(ttt.current_player, 'X')
        self.assertTrue(ttt.make_move(0, 0))
        self.assertTrue(ttt.make_move(0, 1))
        self.assertTrue(ttt.make_move(1, 1))
        self.assertTrue(ttt.make_move(1, 2))
        self.assertEqual(ttt.current_player, 'X')

    def test_make_move_5(self):
        ttt = TicTacToe()
        self.assertEqual(ttt.current_player, 'X')
        self.assertTrue(ttt.make_move(0, 0))
        self.assertTrue(ttt.make_move(0, 1))
        self.assertTrue(ttt.make_move(1, 1))
        self.assertTrue(ttt.make_move(1, 2))
        self.assertTrue(ttt.make_move(2, 2))
        self.assertEqual(ttt.current_player, 'O')


class TicTacToeTestCheckWinner(unittest.TestCase):
    # rows
    def test_check_winner_1(self):
        ttt = TicTacToe()
        moves = [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2)]
        for move in moves:
            ttt.make_move(move[0], move[1])
        self.assertEqual(ttt.check_winner(), 'X')

    # columns
    def test_check_winner_2(self):
        ttt = TicTacToe()
        moves = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
        for move in moves:
            ttt.make_move(move[0], move[1])
        self.assertEqual(ttt.check_winner(), 'X')

    # main diagonals 
    def test_check_winner_3(self):
        ttt = TicTacToe()
        moves = [(0, 0), (0, 1), (1, 1), (0, 2), (2, 2)]
        for move in moves:
            ttt.make_move(move[0], move[1])
        self.assertEqual(ttt.check_winner(), 'X')

    # secondary diagonals 
    def test_check_winner_4(self):
        ttt = TicTacToe()
        moves = [(0, 2), (0, 1), (1, 1), (1, 0), (2, 0)]
        for move in moves:
            ttt.make_move(move[0], move[1])
        self.assertEqual(ttt.check_winner(), 'X')

    def test_check_winner_5(self):
        ttt = TicTacToe()
        moves = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)]
        for move in moves:
            ttt.make_move(move[0], move[1])
        self.assertEqual(ttt.check_winner(), None)


class TicTacToeTestIsBoardFull(unittest.TestCase):
    # not full
    def test_is_board_full_1(self):
        ttt = TicTacToe()
        self.assertFalse(ttt.is_board_full())

    # full
    def test_is_board_full_2(self):
        ttt = TicTacToe()
        moves = [(1, 1), (0, 2), (2, 2), (0, 0), (0, 1), (2, 1), (1, 0), (1, 2), (2, 0)]
        for move in moves:
            ttt.make_move(move[0], move[1])
        self.assertTrue(ttt.is_board_full())

    def test_is_board_full_3(self):
        ttt = TicTacToe()
        moves = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)]
        for move in moves:
            ttt.make_move(move[0], move[1])
        self.assertFalse(ttt.is_board_full())

    def test_is_board_full_4(self):
        ttt = TicTacToe()
        moves = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (0, 2), (1, 2), (2, 1), (2, 2)]
        for move in moves:
            ttt.make_move(move[0], move[1])
        self.assertTrue(ttt.is_board_full())

    def test_is_board_full_5(self):
        ttt = TicTacToe()
        moves = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (0, 2), (1, 2), (2, 1)]
        for move in moves:
            ttt.make_move(move[0], move[1])
        self.assertFalse(ttt.is_board_full())


class TicTacToeTestMain(unittest.TestCase):
    def test_main(self):
        # A draw down way
        ttt = TicTacToe()
        moves = [(1, 1), (0, 2), (2, 2), (0, 0), (0, 1), (2, 1), (1, 0), (1, 2), (2, 0)]
        for move in moves:
            self.assertTrue(ttt.make_move(move[0], move[1]))
            # no winner in this case
            self.assertFalse(ttt.check_winner())
            if move != (2, 0):
                self.assertFalse(ttt.is_board_full())
        self.assertTrue(ttt.is_board_full())

