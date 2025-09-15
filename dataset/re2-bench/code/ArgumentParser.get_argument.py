

class ArgumentParser():

    def __init__(self):
        self.arguments = {}
        self.required = set()
        self.types = {}

    def get_argument(self, key):
        return self.arguments.get(key)
