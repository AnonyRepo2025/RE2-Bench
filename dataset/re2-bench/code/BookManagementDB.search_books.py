
import sqlite3

class BookManagementDB():

    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def search_books(self):
        self.cursor.execute('\n            SELECT * FROM books\n        ')
        books = self.cursor.fetchall()
        return books
