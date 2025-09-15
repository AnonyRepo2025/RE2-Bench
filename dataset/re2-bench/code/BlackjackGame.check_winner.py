
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

    def check_winner(self, player_hand, dealer_hand):
        player_value = self.calculate_hand_value(player_hand)
        dealer_value = self.calculate_hand_value(dealer_hand)
        if ((player_value > 21) and (dealer_value > 21)):
            if (player_value <= dealer_value):
                return 'Player wins'
            else:
                return 'Dealer wins'
        elif (player_value > 21):
            return 'Dealer wins'
        elif (dealer_value > 21):
            return 'Player wins'
        elif (player_value <= dealer_value):
            return 'Dealer wins'
        else:
            return 'Player wins'
