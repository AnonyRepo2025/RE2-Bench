

class Server():

    def __init__(self):
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    def recv(self, info):
        if ((not isinstance(info, dict)) or ('addr' not in info) or ('content' not in info)):
            return (- 1)
        addr = info['addr']
        content = info['content']
        if (addr not in self.white_list):
            return False
        else:
            self.receive_struct = {'addr': addr, 'content': content}
            return self.receive_struct['content']
