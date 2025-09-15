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
       jsonl_path = json_base + "/PushBoxGame.jsonl"
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
# This class implements a functionality of a sokoban game, where the player needs to move boxes to designated targets in order to win.

class PushBoxGame:
    def __init__(self, map):
        """
        Initialize the push box game with the map and various attributes.
        :param map: list[str], the map of the push box game, represented as a list of strings. 
            Each character on the map represents a different element, including the following:
            - '#' represents a wall that neither the player nor the box can pass through;
            - 'O' represents the initial position of the player;
            - 'G' represents the target position;
            - 'X' represents the initial position of the box.
        >>> map = ["#####", "#O  #", "# X #", "#  G#", "#####"]   
        >>> game = PushBoxGame(map)                
        """
        self.map = map
        self.player_row = 0
        self.player_col = 0
        self.targets = []
        self.boxes = []
        self.target_count = 0
        self.is_game_over = False
        self.init_game()

    def init_game(self):
        """
        Initialize the game by setting the positions of the player, targets, and boxes based on the map.
        >>> game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"]) 
        >>> game.targets
        [(3, 3)]
        >>> game.boxes
        [(2, 2)]
        >>> game.player_row
        1
        >>> game.player_col
        1
        """

    def check_win(self):
        """
        Check if the game is won. The game is won when all the boxes are placed on target positions.
        And update the value of self.is_game_over.
        :return self.is_game_over: True if all the boxes are placed on target positions, or False otherwise.
        >>> game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"]) 
        >>> game.check_win()
        """


    def move(self, direction):
        """
        Move the player based on the specified direction and check if the game is won.
        :param direction: str, the direction of the player's movement. 
            It can be 'w', 's', 'a', or 'd' representing up, down, left, or right respectively.

        :return: True if the game is won, False otherwise.
        >>> game = PushBoxGame(["#####", "#O  #", "# X #", "#  G#", "#####"])       
        >>> game.print_map()
        # # # # # 
        # O     #
        #   X   #
        #     G #
        # # # # #
        >>> game.move('d')
        False
        >>> game.move('s')   
        False
        >>> game.move('a')   
        False
        >>> game.move('s') 
        False
        >>> game.move('d') 
        True
        """
'''


class PushBoxGame:
    def __init__(self, map):
        self.map = map
        self.player_row = 0
        self.player_col = 0
        self.targets = []
        self.boxes = []
        self.target_count = 0
        self.is_game_over = False

        self.init_game()

    @inspect_code
    def init_game(self):
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                if self.map[row][col] == "O":
                    self.player_row = row
                    self.player_col = col
                elif self.map[row][col] == "G":
                    self.targets.append((row, col))
                    self.target_count += 1
                elif self.map[row][col] == "X":
                    self.boxes.append((row, col))

    @inspect_code
    def check_win(self):
        box_on_target_count = 0
        for box in self.boxes:
            if box in self.targets:
                box_on_target_count += 1
        if box_on_target_count == self.target_count:
            self.is_game_over = True
        return self.is_game_over

    @inspect_code
    def move(self, direction):
        new_player_row = self.player_row
        new_player_col = self.player_col

        if direction == "w":
            new_player_row -= 1
        elif direction == "s":
            new_player_row += 1
        elif direction == "a":
            new_player_col -= 1
        elif direction == "d":
            new_player_col += 1

        if self.map[new_player_row][new_player_col] != "#":
            if (new_player_row, new_player_col) in self.boxes:
                new_box_row = new_player_row + (new_player_row - self.player_row)
                new_box_col = new_player_col + (new_player_col - self.player_col)

                if self.map[new_box_row][new_box_col] != "#":
                    self.boxes.remove((new_player_row, new_player_col))
                    self.boxes.append((new_box_row, new_box_col))
                    self.player_row = new_player_row
                    self.player_col = new_player_col
            else:
                self.player_row = new_player_row
                self.player_col = new_player_col

        return self.check_win()

import unittest


class PushBoxGameTestInitGame(unittest.TestCase):
    def setUp(self) -> None:
        self.game_map = [
            "#####",
            "#O  #",
            "# X #",
            "#  G#",
            "#####"
        ]
        self.game = PushBoxGame(self.game_map)

    def test_init_game_1(self):
        self.assertEqual(self.game.map, self.game_map)

    def test_init_game_2(self):
        self.assertEqual(self.game.is_game_over, False)

    def test_init_game_3(self):
        self.assertEqual(self.game.player_col, 1)

    def test_init_game_4(self):
        self.assertEqual(self.game.player_row, 1)

    def test_init_game_5(self):
        self.assertEqual(self.game.targets, [(3, 3)])

    def test_init_game_6(self):
        self.assertEqual(self.game.boxes, [(2, 2)])

    def test_init_game_7(self):
        self.assertEqual(self.game.target_count, 1)


class PushBoxGameTestCheckWin(unittest.TestCase):
    def setUp(self) -> None:
        self.game_map = [
            "#####",
            "#O  #",
            "# X #",
            "#  G#",
            "#####"
        ]
        self.game = PushBoxGame(self.game_map)

    def test_check_win_1(self):
        self.assertFalse(self.game.check_win())

    def test_check_win_2(self):
        moves = ['d', 's', 'a', 's', 'd']
        for move in moves:
            self.game.move(move)
        self.assertTrue(self.game.check_win())

class PushBoxGameTestMove(unittest.TestCase):
    def setUp(self) -> None:
        self.game_map = [
            "#####",
            "#O  #",
            "# X #",
            "#  G#",
            "#####"
        ]
        self.game = PushBoxGame(self.game_map)

    def test_move_1(self):
        moves = ['d', 's', 'a', 's']
        for move in moves:
            self.assertFalse(self.game.move(move))
        self.assertTrue(self.game.move('d'))

    def test_move_2(self):
        self.game.move('a')
        self.assertEqual(self.game.player_col, 1)
        self.assertEqual(self.game.player_row, 1)
        self.assertFalse(self.game.is_game_over)

    def test_move_3(self):
        self.game.move('d')
        self.assertEqual(self.game.player_col, 2)
        self.assertEqual(self.game.player_row, 1)
        self.assertFalse(self.game.is_game_over)

    def test_move_4(self):
        self.game.move('s')
        self.assertEqual(self.game.player_col, 1)
        self.assertEqual(self.game.player_row, 2)
        self.assertFalse(self.game.is_game_over)

    def test_move_5(self):
        self.game.move('w')
        self.assertEqual(self.game.player_col, 1)
        self.assertEqual(self.game.player_row, 1)
        self.assertFalse(self.game.is_game_over)

    def test_move_6(self):
        self.game.move('?')
        self.assertFalse(self.game.is_game_over)

    def test_move_7(self):
        self.game_map = [
            "#####",
            "# X #",
            "# O #",
            "#  G#",
            "#####"
        ]
        self.game = PushBoxGame(self.game_map)
        self.game.move('w')
        self.assertEqual(self.game.player_col, 2)
        self.assertEqual(self.game.player_row, 2)
        self.assertFalse(self.game.is_game_over)

