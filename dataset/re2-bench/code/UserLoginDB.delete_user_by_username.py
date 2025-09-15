
import sqlite3

class UserLoginDB():

    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def delete_user_by_username(self, username):
        self.cursor.execute('\n            DELETE FROM users WHERE username = ?\n        ', (username,))
        self.connection.commit()
