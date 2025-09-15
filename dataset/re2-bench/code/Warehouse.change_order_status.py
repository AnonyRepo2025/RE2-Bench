

class Warehouse():

    def __init__(self):
        self.inventory = {}
        self.orders = {}

    def change_order_status(self, order_id, status):
        if (order_id in self.orders):
            self.orders[order_id]['status'] = status
        else:
            return False
