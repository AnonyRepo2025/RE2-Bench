
from datetime import datetime

class EmailClient():

    def __init__(self, addr, capacity) -> None:
        self.addr = addr
        self.capacity = capacity
        self.inbox = []

    def get_occupied_size(self):
        occupied_size = 0
        for email in self.inbox:
            occupied_size += email['size']
        return occupied_size
