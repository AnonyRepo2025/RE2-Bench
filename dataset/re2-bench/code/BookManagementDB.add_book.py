
import sqlite3

class BookManagementDB():

    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def add_book(self, title, author):
        self.cursor.execute('\n            INSERT INTO books (title, author, available)\n            VALUES (?, ?, 1)\n        ', (title, author))
        self.connection.commit()
