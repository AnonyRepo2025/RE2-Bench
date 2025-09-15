

class Server():

    def __init__(self):
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    def del_white_list(self, addr):
        if (addr not in self.white_list):
            return False
        else:
            self.white_list.remove(addr)
            return self.white_list
