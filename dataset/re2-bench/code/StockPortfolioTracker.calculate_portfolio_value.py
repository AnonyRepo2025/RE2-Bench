

class StockPortfolioTracker():

    def __init__(self, cash_balance):
        self.portfolio = []
        self.cash_balance = cash_balance

    def calculate_portfolio_value(self):
        total_value = self.cash_balance
        for stock in self.portfolio:
            total_value += (stock['price'] * stock['quantity'])
        return total_value
