

class Order():

    def __init__(self):
        self.menu = []
        self.selected_dishes = []
        self.sales = {}

    def calculate_total(self):
        total = 0
        for dish in self.selected_dishes:
            total += ((dish['price'] * dish['count']) * self.sales[dish['dish']])
        return total

    def checkout(self):
        if (len(self.selected_dishes) == 0):
            return False
        total = self.calculate_total()
        self.selected_dishes = []
        return total
