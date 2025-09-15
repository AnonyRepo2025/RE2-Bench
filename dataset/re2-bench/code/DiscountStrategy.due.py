

class DiscountStrategy():

    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
        self.__total = self.total()

    def due(self):
        if (self.promotion is None):
            discount = 0
        else:
            discount = self.promotion(self)
        return (self.__total - discount)
