

class ArgumentParser():

    def __init__(self):
        self.arguments = {}
        self.required = set()
        self.types = {}

    def add_argument(self, arg, required=False, arg_type=str):
        if required:
            self.required.add(arg)
        self.types[arg] = arg_type
