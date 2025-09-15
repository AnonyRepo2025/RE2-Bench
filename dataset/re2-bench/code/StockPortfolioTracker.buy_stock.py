

class StockPortfolioTracker():

    def __init__(self, cash_balance):
        self.portfolio = []
        self.cash_balance = cash_balance

    def add_stock(self, stock):
        for pf in self.portfolio:
            if (pf['name'] == stock['name']):
                pf['quantity'] += stock['quantity']
                return
        self.portfolio.append(stock)

    def buy_stock(self, stock):
        if ((stock['price'] * stock['quantity']) > self.cash_balance):
            return False
        else:
            self.add_stock(stock)
            self.cash_balance -= (stock['price'] * stock['quantity'])
            return True
