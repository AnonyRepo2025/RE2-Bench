
from datetime import datetime

class Chat():

    def __init__(self):
        self.users = {}

    def get_messages(self, username):
        if (username not in self.users):
            return []
        return self.users[username]
