

class DiscountStrategy():

    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
        self.__total = self.total()

    @staticmethod
    def FidelityPromo(order):
        return ((order.total() * 0.05) if (order.customer['fidelity'] >= 1000) else 0)
