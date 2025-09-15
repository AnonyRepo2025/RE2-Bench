
import sqlite3

class UserLoginDB():

    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def insert_user(self, username, password):
        self.cursor.execute('\n            INSERT INTO users (username, password)\n            VALUES (?, ?)\n        ', (username, password))
        self.connection.commit()
