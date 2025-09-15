

class StockPortfolioTracker():

    def __init__(self, cash_balance):
        self.portfolio = []
        self.cash_balance = cash_balance

    def get_stock_value(self, stock):
        return (stock['price'] * stock['quantity'])
