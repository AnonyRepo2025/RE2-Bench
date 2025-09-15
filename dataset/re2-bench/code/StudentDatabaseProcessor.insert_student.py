
import sqlite3

class StudentDatabaseProcessor():

    def __init__(self, database_name):
        self.database_name = database_name

    def insert_student(self, student_data):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        insert_query = '\n        INSERT INTO students (name, age, gender, grade)\n        VALUES (?, ?, ?, ?)\n        '
        cursor.execute(insert_query, (student_data['name'], student_data['age'], student_data['gender'], student_data['grade']))
        conn.commit()
        conn.close()
