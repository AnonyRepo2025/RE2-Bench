
import sqlite3

class MovieTicketDB():

    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def search_tickets_by_customer(self, customer_name):
        self.cursor.execute('\n            SELECT * FROM tickets WHERE customer_name = ?\n        ', (customer_name,))
        tickets = self.cursor.fetchall()
        return tickets
