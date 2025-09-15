

class Server():

    def __init__(self):
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    def add_white_list(self, addr):
        if (addr in self.white_list):
            return False
        else:
            self.white_list.append(addr)
            return self.white_list
