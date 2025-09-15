
import sqlite3

class MovieTicketDB():

    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def insert_ticket(self, movie_name, theater_name, seat_number, customer_name):
        self.cursor.execute('\n            INSERT INTO tickets (movie_name, theater_name, seat_number, customer_name)\n            VALUES (?, ?, ?, ?)\n        ', (movie_name, theater_name, seat_number, customer_name))
        self.connection.commit()
