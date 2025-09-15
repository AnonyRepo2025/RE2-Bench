

class CurrencyConverter():

    def __init__(self):
        self.rates = {'USD': 1.0, 'EUR': 0.85, 'GBP': 0.72, 'JPY': 110.15, 'CAD': 1.23, 'AUD': 1.34, 'CNY': 6.4}

    def get_supported_currencies(self):
        return list(self.rates.keys())
