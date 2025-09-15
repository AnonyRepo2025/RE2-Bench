
import sqlite3

class BookManagementDB():

    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('\n            CREATE TABLE IF NOT EXISTS books (\n                id INTEGER PRIMARY KEY,\n                title TEXT,\n                author TEXT,\n                available INTEGER\n            )\n        ')
        self.connection.commit()
