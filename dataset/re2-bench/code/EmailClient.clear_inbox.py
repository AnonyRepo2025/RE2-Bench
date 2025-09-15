
from datetime import datetime

class EmailClient():

    def __init__(self, addr, capacity) -> None:
        self.addr = addr
        self.capacity = capacity
        self.inbox = []

    def clear_inbox(self, size):
        if (len(self.addr) == 0):
            return
        freed_space = 0
        while ((freed_space < size) and self.inbox):
            email = self.inbox[0]
            freed_space += email['size']
            del self.inbox[0]
