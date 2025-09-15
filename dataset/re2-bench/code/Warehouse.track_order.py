

class Warehouse():

    def __init__(self):
        self.inventory = {}
        self.orders = {}

    def track_order(self, order_id):
        if (order_id in self.orders):
            return self.orders[order_id]['status']
        else:
            return False
