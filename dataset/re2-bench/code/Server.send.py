

class Server():

    def __init__(self):
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    def send(self, info):
        if ((not isinstance(info, dict)) or ('addr' not in info) or ('content' not in info)):
            return 'info structure is not correct'
        self.send_struct = {'addr': info['addr'], 'content': info['content']}
