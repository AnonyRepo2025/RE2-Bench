
import sqlite3

class BookManagementDB():

    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def borrow_book(self, book_id):
        self.cursor.execute('\n            UPDATE books SET available = 0 WHERE id = ?\n        ', (book_id,))
        self.connection.commit()
