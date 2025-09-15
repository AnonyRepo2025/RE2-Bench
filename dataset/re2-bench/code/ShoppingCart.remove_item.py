

class ShoppingCart():

    def __init__(self):
        self.items = {}

    def remove_item(self, item, quantity=1):
        if (item in self.items):
            self.items[item]['quantity'] -= quantity
        else:
            pass
