
import sqlite3
import pandas as pd

class DatabaseProcessor():

    def __init__(self, database_name):
        self.database_name = database_name

    def insert_into_database(self, table_name, data):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        for item in data:
            insert_query = f'INSERT INTO {table_name} (name, age) VALUES (?, ?)'
            cursor.execute(insert_query, (item['name'], item['age']))
        conn.commit()
        conn.close()
