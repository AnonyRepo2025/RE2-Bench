
from datetime import datetime

class EmailClient():

    def __init__(self, addr, capacity) -> None:
        self.addr = addr
        self.capacity = capacity
        self.inbox = []

    def send_to(self, recv, content, size):
        if (not recv.is_full_with_one_more_email(size)):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            email = {'sender': self.addr, 'receiver': recv.addr, 'content': content, 'size': size, 'time': timestamp, 'state': 'unread'}
            recv.inbox.append(email)
            return True
        else:
            self.clear_inbox(size)
            return False

    def clear_inbox(self, size):
        if (len(self.addr) == 0):
            return
        freed_space = 0
        while ((freed_space < size) and self.inbox):
            email = self.inbox[0]
            freed_space += email['size']
            del self.inbox[0]
