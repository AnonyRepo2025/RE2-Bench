

class DiscountStrategy():

    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
        self.__total = self.total()

    def total(self):
        self.__total = sum(((item['quantity'] * item['price']) for item in self.cart))
        return self.__total
