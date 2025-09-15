

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
