

class DiscountStrategy():

    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
        self.__total = self.total()

    @staticmethod
    def BulkItemPromo(order):
        discount = 0
        for item in order.cart:
            if (item['quantity'] >= 20):
                discount += ((item['quantity'] * item['price']) * 0.1)
        return discount
