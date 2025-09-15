

class SignInSystem():

    def __init__(self):
        self.users = {}

    def all_signed_in(self):
        if all(self.users.values()):
            return True
        else:
            return False
