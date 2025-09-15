

class Warehouse():

    def __init__(self):
        self.inventory = {}
        self.orders = {}

    def get_product_quantity(self, product_id):
        if (product_id in self.inventory):
            return self.inventory[product_id]['quantity']
        else:
            return False
