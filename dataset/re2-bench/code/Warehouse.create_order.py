

class Warehouse():

    def __init__(self):
        self.inventory = {}
        self.orders = {}

    def update_product_quantity(self, product_id, quantity):
        if (product_id in self.inventory):
            self.inventory[product_id]['quantity'] += quantity

    def get_product_quantity(self, product_id):
        if (product_id in self.inventory):
            return self.inventory[product_id]['quantity']
        else:
            return False

    def create_order(self, order_id, product_id, quantity):
        if (self.get_product_quantity(product_id) >= quantity):
            self.update_product_quantity(product_id, (- quantity))
            self.orders[order_id] = {'product_id': product_id, 'quantity': quantity, 'status': 'Shipped'}
        else:
            return False
