

class ArgumentParser():

    def __init__(self):
        self.arguments = {}
        self.required = set()
        self.types = {}

    def _convert_type(self, arg, value):
        try:
            return self.types[arg](value)
        except (ValueError, KeyError):
            return value
