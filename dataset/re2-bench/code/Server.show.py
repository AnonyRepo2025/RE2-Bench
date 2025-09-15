

class Server():

    def __init__(self):
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    def show(self, type):
        if (type == 'send'):
            return self.send_struct
        elif (type == 'receive'):
            return self.receive_struct
        else:
            return False
