

class Warehouse():

    def __init__(self):
        self.inventory = {}
        self.orders = {}

    def update_product_quantity(self, product_id, quantity):
        if (product_id in self.inventory):
            self.inventory[product_id]['quantity'] += quantity
