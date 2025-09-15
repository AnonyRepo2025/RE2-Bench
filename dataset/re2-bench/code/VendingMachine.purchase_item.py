

class VendingMachine():

    def __init__(self):
        self.inventory = {}
        self.balance = 0

    def purchase_item(self, item_name):
        if (item_name in self.inventory):
            item = self.inventory[item_name]
            if ((item['quantity'] > 0) and (self.balance >= item['price'])):
                self.balance -= item['price']
                item['quantity'] -= 1
                return self.balance
            else:
                return False
        else:
            return False
