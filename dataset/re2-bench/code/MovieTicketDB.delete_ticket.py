
import sqlite3

class MovieTicketDB():

    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def delete_ticket(self, ticket_id):
        self.cursor.execute('\n            DELETE FROM tickets WHERE id = ?\n        ', (ticket_id,))
        self.connection.commit()
