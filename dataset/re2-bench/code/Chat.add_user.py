
from datetime import datetime

class Chat():

    def __init__(self):
        self.users = {}

    def add_user(self, username):
        if (username in self.users):
            return False
        else:
            self.users[username] = []
            return True
