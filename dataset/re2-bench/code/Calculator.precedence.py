

class Calculator():

    def __init__(self):
        self.operators = {'+': (lambda x, y: (x + y)), '-': (lambda x, y: (x - y)), '*': (lambda x, y: (x * y)), '/': (lambda x, y: (x / y)), '^': (lambda x, y: (x ** y))}

    def precedence(self, operator):
        precedences = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
        return precedences.get(operator, 0)
