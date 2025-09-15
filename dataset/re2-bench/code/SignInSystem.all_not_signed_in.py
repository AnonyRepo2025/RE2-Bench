

class SignInSystem():

    def __init__(self):
        self.users = {}

    def all_not_signed_in(self):
        not_signed_in_users = []
        for (username, signed_in) in self.users.items():
            if (not signed_in):
                not_signed_in_users.append(username)
        return not_signed_in_users
