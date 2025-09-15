
import sqlite3

class MovieTicketDB():

    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('\n            CREATE TABLE IF NOT EXISTS tickets (\n                id INTEGER PRIMARY KEY,\n                movie_name TEXT,\n                theater_name TEXT,\n                seat_number TEXT,\n                customer_name TEXT\n            )\n        ')
        self.connection.commit()
