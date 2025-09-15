import inspect
import json
import os
from datetime import datetime

def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def recursive_object_seralizer(obj, visited):
    seralized_dict = {}
    keys = list(obj.__dict__)
    for k in keys:
        if id(obj.__dict__[k]) in visited:
            seralized_dict[k] = "<RECURSIVE {}>".format(obj.__dict__[k])
            continue
        if isinstance(obj.__dict__[k], (float, int, str, bool, type(None))):
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], tuple):
            ## handle tuple
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], set):
            ## handle set
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], list):
            ## handle list
            seralized_dict[k] = obj.__dict__[k]
        elif hasattr(obj.__dict__[k], '__dict__'):
            ## handle object
            visited.append(id(obj.__dict__[k]))
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], dict):
            visited.append(id(obj.__dict__[k]))
            seralized_dict[k] = obj.__dict__[k]
        elif callable(obj.__dict__[k]):
            ## handle function
            if hasattr(obj.__dict__[k], '__name__'):
                seralized_dict[k] = "<function {}>".format(obj.__dict__[k].__name__)
        else:
            seralized_dict[k] = str(obj.__dict__[k])
    return seralized_dict

def inspect_code(func):
   def wrapper(*args, **kwargs):
       visited = []
       json_base = "/home/changshu/ClassEval/data/benchmark_solution_code/input-output/"
       if not os.path.exists(json_base):
           os.mkdir(json_base)
       jsonl_path = json_base + "/StudentDatabaseProcessor.jsonl"
       para_dict = {"name": func.__name__}
       args_names = inspect.getfullargspec(func).args
       if len(args) > 0 and hasattr(args[0], '__dict__') and args_names[0] == 'self':
           ## 'self'
           self_args = args[0]
           para_dict['self'] = recursive_object_seralizer(self_args, [id(self_args)])
       else:
           para_dict['self'] = {}
       if len(args) > 0 :
           if args_names[0] == 'self':
               other_args = {}
               for m,n in zip(args_names[1:], args[1:]):
                   other_args[m] = n
           else:
               other_args = {}
               for m,n in zip(args_names, args):
                   other_args[m] = n
           
           para_dict['args'] = other_args
       else:
           para_dict['args'] = {}
       if kwargs:
           para_dict['kwargs'] = kwargs
       else:
           para_dict['kwargs'] = {}
          
       result = func(*args, **kwargs)
       para_dict["return"] = result
       with open(jsonl_path, 'a') as f:
           f.write(json.dumps(para_dict, default=custom_serializer) + "\n")
       return result
   return wrapper


'''
# This is a class with database operation, including inserting student information, searching for student information by name, and deleting student information by name.

import sqlite3

class StudentDatabaseProcessor:
    def __init__(self, database_name):
        """
        Initializes the StudentDatabaseProcessor object with the specified database name.
        :param database_name: str, the name of the SQLite database.
        """
        self.database_name = database_name

    def create_student_table(self):
        """
        Creates a "students" table in the database if it does not exist already.Fields include ID of type int, name of type str, age of type int, gender of type str, and grade of type int
        :return: None
        >>> processor = StudentDatabaseProcessor("students.db")
        >>> processor.create_student_table()
        """

    def insert_student(self, student_data):
        """
        Inserts a new student into the "students" table.
        :param student_data: dict, a dictionary containing the student's information (name, age, gender, grade).
        :return: None
        >>> processor = StudentDatabaseProcessor("students.db")
        >>> processor.create_student_table()
        >>> student_data = {'name': 'John', 'age': 15, 'gender': 'Male', 'grade': 9}
        >>> processor.insert_student(student_data)
        """

    def search_student_by_name(self, name):
        """
        Searches for a student in the "students" table by their name.
        :param name: str, the name of the student to search for.
        :return: list of tuples, the rows from the "students" table that match the search criteria.
        >>> processor = StudentDatabaseProcessor("students.db")
        >>> processor.create_student_table()
        >>> result = processor.search_student_by_name("John")
        """

    def delete_student_by_name(self, name):
        """
        Deletes a student from the "students" table by their name.
        :param name: str, the name of the student to delete.
        :return: None
        >>> processor = StudentDatabaseProcessor("students.db")
        >>> processor.create_student_table()
        >>> student_data = {'name': 'John', 'age': 15, 'gender': 'Male', 'grade': 9}
        >>> processor.insert_student(student_data)
        >>> processor.delete_student_by_name("John")
        """
'''

import sqlite3


class StudentDatabaseProcessor:

    def __init__(self, database_name):
        self.database_name = database_name

    @inspect_code
    def create_student_table(self):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()

        create_table_query = """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            gender TEXT,
            grade INTEGER
        )
        """
        cursor.execute(create_table_query)

        conn.commit()
        conn.close()

    @inspect_code
    def insert_student(self, student_data):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO students (name, age, gender, grade)
        VALUES (?, ?, ?, ?)
        """
        cursor.execute(insert_query,
                       (student_data['name'], student_data['age'], student_data['gender'], student_data['grade']))

        conn.commit()
        conn.close()

    @inspect_code
    def search_student_by_name(self, name):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()

        select_query = "SELECT * FROM students WHERE name = ?"
        cursor.execute(select_query, (name,))
        result = cursor.fetchall()

        conn.close()

        return result

    @inspect_code
    def delete_student_by_name(self, name):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()

        delete_query = "DELETE FROM students WHERE name = ?"
        cursor.execute(delete_query, (name,))

        conn.commit()
        conn.close()



import unittest


class StudentDatabaseProcessorTestInsertStudent(unittest.TestCase):
    def setUp(self):
        self.processor = StudentDatabaseProcessor("test_database.db")
        self.processor.create_student_table()

    def tearDown(self):
        conn = sqlite3.connect("test_database.db")
        conn.execute("DROP TABLE IF EXISTS students")
        conn.commit()
        conn.close()

    def test_insert_student_1(self):
        student_data = {
            'name': 'Alice',
            'age': 20,
            'gender': 'female',
            'grade': 90
        }
        self.processor.insert_student(student_data)

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('Alice',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'Alice')

    def test_insert_student_2(self):
        student_data = {
            'name': 'aaa',
            'age': 20,
            'gender': 'female',
            'grade': 90
        }
        self.processor.insert_student(student_data)

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('aaa',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'aaa')

    def test_insert_student_3(self):
        student_data = {
            'name': 'bbb',
            'age': 20,
            'gender': 'female',
            'grade': 90
        }
        self.processor.insert_student(student_data)

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('bbb',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'bbb')

    def test_insert_student_4(self):
        student_data = {
            'name': 'ccc',
            'age': 20,
            'gender': 'female',
            'grade': 90
        }
        self.processor.insert_student(student_data)

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('ccc',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'ccc')

    def test_insert_student_5(self):
        student_data = {
            'name': 'ddd',
            'age': 20,
            'gender': 'female',
            'grade': 90
        }
        self.processor.insert_student(student_data)

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('ddd',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'ddd')


class StudentDatabaseProcessorTestSearchStudentByName(unittest.TestCase):
    def setUp(self):
        self.processor = StudentDatabaseProcessor("test_database.db")
        self.processor.create_student_table()

    def tearDown(self):
        conn = sqlite3.connect("test_database.db")
        conn.execute("DROP TABLE IF EXISTS students")
        conn.commit()
        conn.close()

    def test_search_student_by_name_1(self):
        student_data = {
            'name': 'Bob',
            'age': 19,
            'gender': 'male',
            'grade': 85
        }
        self.processor.insert_student(student_data)

        result = self.processor.search_student_by_name('Bob')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'Bob')

    def test_search_student_by_name_2(self):
        student_data = {
            'name': 'aaa',
            'age': 19,
            'gender': 'male',
            'grade': 85
        }
        self.processor.insert_student(student_data)

        result = self.processor.search_student_by_name('aaa')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'aaa')

    def test_search_student_by_name_3(self):
        student_data = {
            'name': 'bbb',
            'age': 19,
            'gender': 'male',
            'grade': 85
        }
        self.processor.insert_student(student_data)

        result = self.processor.search_student_by_name('bbb')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'bbb')

    def test_search_student_by_name_4(self):
        student_data = {
            'name': 'ccc',
            'age': 19,
            'gender': 'male',
            'grade': 85
        }
        self.processor.insert_student(student_data)

        result = self.processor.search_student_by_name('ccc')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'ccc')

    def test_search_student_by_name_5(self):
        student_data = {
            'name': 'ddd',
            'age': 19,
            'gender': 'male',
            'grade': 85
        }
        self.processor.insert_student(student_data)

        result = self.processor.search_student_by_name('ddd')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'ddd')


class StudentDatabaseProcessorTestDeleteStudentByName(unittest.TestCase):
    def setUp(self):
        self.processor = StudentDatabaseProcessor("test_database.db")
        self.processor.create_student_table()

    def tearDown(self):
        conn = sqlite3.connect("test_database.db")
        conn.execute("DROP TABLE IF EXISTS students")
        conn.commit()
        conn.close()

    def test_delete_student_by_name_1(self):
        student_data = {
            'name': 'Charlie',
            'age': 18,
            'gender': 'male',
            'grade': 95
        }
        self.processor.insert_student(student_data)

        self.processor.delete_student_by_name('Charlie')

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('Charlie',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 0)

    def test_delete_student_by_name_2(self):
        student_data = {
            'name': 'aaa',
            'age': 18,
            'gender': 'male',
            'grade': 95
        }
        self.processor.insert_student(student_data)

        self.processor.delete_student_by_name('aaa')

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('aaa',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 0)

    def test_delete_student_by_name_3(self):
        student_data = {
            'name': 'bbb',
            'age': 18,
            'gender': 'male',
            'grade': 95
        }
        self.processor.insert_student(student_data)

        self.processor.delete_student_by_name('bbb')

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('bbb',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 0)

    def test_delete_student_by_name_4(self):
        student_data = {
            'name': 'ccc',
            'age': 18,
            'gender': 'male',
            'grade': 95
        }
        self.processor.insert_student(student_data)

        self.processor.delete_student_by_name('ccc')

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('ccc',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 0)

    def test_delete_student_by_name_5(self):
        student_data = {
            'name': 'ddd',
            'age': 18,
            'gender': 'male',
            'grade': 95
        }
        self.processor.insert_student(student_data)

        self.processor.delete_student_by_name('ddd')

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('ddd',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 0)


class StudentDatabaseProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = StudentDatabaseProcessor("test_database.db")
        self.processor.create_student_table()

    def tearDown(self):
        conn = sqlite3.connect("test_database.db")
        conn.execute("DROP TABLE IF EXISTS students")
        conn.commit()
        conn.close()

    def test_StudentDatabaseProcessor(self):
        student_data = {
            'name': 'Alice',
            'age': 20,
            'gender': 'female',
            'grade': 90
        }
        self.processor.insert_student(student_data)

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('Alice',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'Alice')

        student_data = {
            'name': 'Bob',
            'age': 19,
            'gender': 'male',
            'grade': 85
        }
        self.processor.insert_student(student_data)

        result = self.processor.search_student_by_name('Bob')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'Bob')

        student_data = {
            'name': 'Charlie',
            'age': 18,
            'gender': 'male',
            'grade': 95
        }
        self.processor.insert_student(student_data)

        self.processor.delete_student_by_name('Charlie')

        conn = sqlite3.connect("test_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE name=?", ('Charlie',))
        result = cursor.fetchall()
        conn.close()

        self.assertEqual(len(result), 0)

