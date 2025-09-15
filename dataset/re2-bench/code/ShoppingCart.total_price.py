

class ShoppingCart():

    def __init__(self):
        self.items = {}

    def total_price(self) -> float:
        return sum([(item['quantity'] * item['price']) for item in self.items.values()])
