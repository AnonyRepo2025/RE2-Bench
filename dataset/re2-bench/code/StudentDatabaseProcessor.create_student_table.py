
import sqlite3

class StudentDatabaseProcessor():

    def __init__(self, database_name):
        self.database_name = database_name

    def create_student_table(self):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        create_table_query = '\n        CREATE TABLE IF NOT EXISTS students (\n            id INTEGER PRIMARY KEY,\n            name TEXT,\n            age INTEGER,\n            gender TEXT,\n            grade INTEGER\n        )\n        '
        cursor.execute(create_table_query)
        conn.commit()
        conn.close()
