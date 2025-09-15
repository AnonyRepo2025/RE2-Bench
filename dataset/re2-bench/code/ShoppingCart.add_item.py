

class ShoppingCart():

    def __init__(self):
        self.items = {}

    def add_item(self, item, price, quantity=1):
        if (item in self.items):
            self.items[item] = {'price': price, 'quantity': quantity}
        else:
            self.items[item] = {'price': price, 'quantity': quantity}
