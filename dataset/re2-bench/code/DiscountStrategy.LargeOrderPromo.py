

class DiscountStrategy():

    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
        self.__total = self.total()

    @staticmethod
    def LargeOrderPromo(order):
        return ((order.total() * 0.07) if (len({item['product'] for item in order.cart}) >= 10) else 0)
