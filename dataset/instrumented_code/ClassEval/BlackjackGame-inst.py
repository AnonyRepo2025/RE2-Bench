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
       jsonl_path = json_base + "/BlackjackGame.jsonl"
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
# This is a class representing a game of blackjack, which includes creating a deck, calculating the value of a hand, and determine the winner based on the hand values of the player and dealer.

import random
class BlackjackGame:
    def __init__(self):
        """
        Initialize the Blackjack Game with the attribute deck, player_hand and dealer_hand.
        While initializing deck attribute, call the create_deck method to generate.
        The deck stores 52 rondom order poker with the Jokers removed, format is ['AS', '2S', ...].
        player_hand is a list which stores player's hand cards.
        dealer_hand is is a list which stores dealer's hand cards.
        """
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []

    def create_deck(self):
        """
        Create a deck of 52 cards, which stores 52 rondom order poker with the Jokers removed.
        :return: a list of 52 rondom order poker with the Jokers removed, format is ['AS', '2S', ...].
        >>> black_jack_game = BlackjackGame()
        >>> black_jack_game.create_deck()
        ['QD', '9D', 'JC', 'QH', '2S', 'JH', '7D', '6H', '9S', '5C', '7H', 'QS', '5H',
        '6C', '7C', '3D', '10C', 'AD', '4C', '5D', 'AH', '2D', 'QC', 'KH', '9C', '9H',
        '4H', 'JS', '6S', '8H', '8C', '4S', '3H', '10H', '7S', '6D', '3C', 'KC', '3S',
        '2H', '10D', 'KS', '4D', 'AC', '10S', '2C', 'KD', '5S', 'JD', '8S', 'AS', '8D']
        """

    def calculate_hand_value(self, hand):
        """
        Calculate the value of the poker cards stored in hand list according to the rules of the Blackjack Game.
        If the card is a digit, its value is added to the total hand value.
        Value of J, Q, or K is 10, while Aces are worth 11.
        If the total hand value exceeds 21 and there are Aces present, one Ace is treated as having a value of 1 instead of 11,
        until the hand value is less than or equal to 21, or all Aces have been counted as value of 1.
        :param hand: list
        :return: the value of the poker cards stored in hand list, a number.
        >>> black_jack_game.calculate_hand_value(['QD', '9D', 'JC', 'QH', 'AS'])
        40
        """

    def check_winner(self, player_hand, dealer_hand):
        """
        Determines the winner of a game by comparing the hand values of the player and dealer.
        rule:
        If both players have hand values that are equal to or less than 21, the winner is the one whose hand value is closer to 21.
        Otherwise, the winner is the one with the lower hand value.
        :param player_hand: list
        :param dealer_hand: list
        :return: the result of the game, only two certain str: 'Dealer wins' or 'Player wins'
        >>> black_jack_game.check_winner(['QD', '9D', 'JC', 'QH', 'AS'], ['QD', '9D', 'JC', 'QH', '2S'])
        'Player wins'
        """

'''

import random


class BlackjackGame:
    def __init__(self):
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []

    @inspect_code
    def create_deck(self):
        deck = []
        suits = ['S', 'C', 'D', 'H']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        for suit in suits:
            for rank in ranks:
                deck.append(rank + suit)
        random.shuffle(deck)
        return deck

    @inspect_code
    def calculate_hand_value(self, hand):
        value = 0
        num_aces = 0
        for card in hand:
            rank = card[:-1]
            if rank.isdigit():
                value += int(rank)
            elif rank in ['J', 'Q', 'K']:
                value += 10
            elif rank == 'A':
                value += 11
                num_aces += 1
        while value > 21 and num_aces > 0:
            value -= 10
            num_aces -= 1
        return value

    @inspect_code
    def check_winner(self, player_hand, dealer_hand):
        player_value = self.calculate_hand_value(player_hand)
        dealer_value = self.calculate_hand_value(dealer_hand)
        if player_value > 21 and dealer_value > 21:
            if player_value <= dealer_value:
                return 'Player wins'
            else:
                return 'Dealer wins'
        elif player_value > 21:
            return 'Dealer wins'
        elif dealer_value > 21:
            return 'Player wins'
        else:
            if player_value <= dealer_value:
                return 'Dealer wins'
            else:
                return 'Player wins'

import unittest

class BlackjackGameTestCreateDeck(unittest.TestCase):
    def setUp(self):
        self.blackjackGame = BlackjackGame()
        self.deck = self.blackjackGame.deck

    def test_create_deck_1(self):
        self.assertEqual(len(self.deck), 52)

    def test_create_deck_2(self):
        suits = ['S', 'C', 'D', 'H']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        for suit in suits:
            for rank in ranks:
                self.assertIn(rank + suit, self.deck)

    def test_create_deck_3(self):
        suits = ['S', 'C', 'D', 'H']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9']
        for suit in suits:
            for rank in ranks:
                self.assertIn(rank + suit, self.deck)

    def test_create_deck_4(self):
        suits = ['S', 'C', 'D', 'H']
        ranks = ['10', 'J', 'Q', 'K']
        for suit in suits:
            for rank in ranks:
                self.assertIn(rank + suit, self.deck)

    def test_create_deck_5(self):
        suits = ['S', 'C', 'D', 'H']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9']
        for suit in suits:
            for rank in ranks:
                self.assertIn(rank + suit, self.deck)

class BlackjackGameTestCalculateHandValue(unittest.TestCase):
    def test_calculate_hand_value_1(self):
        blackjackGame = BlackjackGame()
        hand = ['2S', '3S', '4S', '5S']
        self.assertEqual(blackjackGame.calculate_hand_value(hand), 14)

    def test_calculate_hand_value_2(self):
        blackjackGame = BlackjackGame()
        hand = ['2S', '3S', 'JS', 'QS']
        self.assertEqual(blackjackGame.calculate_hand_value(hand), 25)

    def test_calculate_hand_value_3(self):
        blackjackGame = BlackjackGame()
        hand = ['2S', '3S', '4S', 'AS']
        self.assertEqual(blackjackGame.calculate_hand_value(hand), 20)

    def test_calculate_hand_value_4(self):
        blackjackGame = BlackjackGame()
        hand = ['JS', 'QS', '4S', 'AS']
        self.assertEqual(blackjackGame.calculate_hand_value(hand), 25)

    def test_calculate_hand_value_5(self):
        blackjackGame = BlackjackGame()
        hand = ['JS', 'QS', 'AS', 'AS', 'AS']
        self.assertEqual(blackjackGame.calculate_hand_value(hand), 23)

    def test_calculate_hand_value_6(self):
        blackjackGame = BlackjackGame()
        hand = ['JS', 'QS', 'BS', 'CS']
        self.assertEqual(blackjackGame.calculate_hand_value(hand), 20)


class BlackjackGameTestCheckWinner(unittest.TestCase):
    def setUp(self):
        self.blackjackGame = BlackjackGame()

    # player > 21 but dealer not, dealer wins.
    def test_check_winner_1(self):
        player_hand = ['2S', 'JS', 'QS']
        dealer_hand = ['7S', '9S']
        self.assertEqual(self.blackjackGame.check_winner(player_hand, dealer_hand), 'Dealer wins')

    # dealer > 21 but player not, player wins.
    def test_check_winner_2(self):
        player_hand = ['2S', '4S', '5S']
        dealer_hand = ['2S', 'JS', 'QS']
        self.assertEqual(self.blackjackGame.check_winner(player_hand, dealer_hand), 'Player wins')

    # both > 21 but dealer smaller, dealer wins.
    def test_check_winner_3(self):
        player_hand = ['3S', 'JS', 'QS']
        dealer_hand = ['2S', 'JS', 'QS']
        self.assertEqual(self.blackjackGame.check_winner(player_hand, dealer_hand), 'Dealer wins')

    # both > 21 but player smaller, player wins.
    def test_check_winner_4(self):
        player_hand = ['2S', 'JS', 'QS']
        dealer_hand = ['3S', 'JS', 'QS']
        self.assertEqual(self.blackjackGame.check_winner(player_hand, dealer_hand), 'Player wins')

    # both < 21 but dealer is bigger, dealer wins.
    def test_check_winner_5(self):
        player_hand = ['2S', '3S', '5S']
        dealer_hand = ['AS', 'JS']
        self.assertEqual(self.blackjackGame.check_winner(player_hand, dealer_hand), 'Dealer wins')

    # both < 21 but player is bigger, player wins.
    def test_check_winner_6(self):
        player_hand = ['AS', 'JS']
        dealer_hand = ['2S', '3S', '5S']
        self.assertEqual(self.blackjackGame.check_winner(player_hand, dealer_hand), 'Player wins')


class BlackjackGameTestMain(unittest.TestCase):
    # calculate_hand_value method will be invoked in check_winner
    def test_main_1(self):
        blackjackGame = BlackjackGame()
        deck = blackjackGame.deck
        suits = ['S', 'C', 'D', 'H']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        for suit in suits:
            for rank in ranks:
                self.assertIn(rank + suit, deck)
        player_hand = ['2S', 'JS', 'QS']
        dealer_hand = ['7S', '9S']
        self.assertEqual(blackjackGame.check_winner(player_hand, dealer_hand), 'Dealer wins')
