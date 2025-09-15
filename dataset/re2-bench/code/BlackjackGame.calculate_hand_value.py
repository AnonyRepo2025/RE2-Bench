
import random

class BlackjackGame():

    def __init__(self):
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []

    def calculate_hand_value(self, hand):
        value = 0
        num_aces = 0
        for card in hand:
            rank = card[:(- 1)]
            if rank.isdigit():
                value += int(rank)
            elif (rank in ['J', 'Q', 'K']):
                value += 10
            elif (rank == 'A'):
                value += 11
                num_aces += 1
        while ((value > 21) and (num_aces > 0)):
            value -= 10
            num_aces -= 1
        return value
