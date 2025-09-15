

class SignInSystem():

    def __init__(self):
        self.users = {}

    def check_sign_in(self, username):
        if (username not in self.users):
            return False
        elif self.users[username]:
            return True
        else:
            return False
