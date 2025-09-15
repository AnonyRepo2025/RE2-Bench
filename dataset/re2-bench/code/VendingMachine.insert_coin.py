

class VendingMachine():

    def __init__(self):
        self.inventory = {}
        self.balance = 0

    def insert_coin(self, amount):
        self.balance += amount
        return self.balance
