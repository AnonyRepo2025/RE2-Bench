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
       jsonl_path = json_base + "/UserLoginDB.jsonl"
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
# This is a database management class for user login verification, providing functions for inserting user information, searching user information, deleting user information, and validating user login.

class UserLoginDB:
    def __init__(self, db_name):
        """
        Initializes the UserLoginDB object with the specified database name.
        :param db_name: str, the name of the SQLite database.
        """
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def insert_user(self, username, password):
        """
        Inserts a new user into the "users" table.
        :param username: str, the username of the user.
        :param password: str, the password of the user.
        :return: None
        >>> user_db = UserLoginDB("user_database.db")
        >>> user_db.create_table()
        >>> user_db.insert_user('user1', 'pass1')
        """


    def search_user_by_username(self, username):
        """
        Searches for users in the "users" table by username.
        :param username: str, the username of the user to search for.
        :return:list of tuples, the rows from the "users" table that match the search criteria.
        >>> user_db = UserLoginDB("user_database.db")
        >>> user_db.create_table()
        >>> user_db.insert_user('user1', 'pass1')
        >>> result = user_db.search_user_by_username('user1')
        len(result) = 1
        """


    def delete_user_by_username(self, username):
        """
        Deletes a user from the "users" table by username.
        :param username: str, the username of the user to delete.
        :return: None
        >>> user_db = UserLoginDB("user_database.db")
        >>> user_db.create_table()
        >>> user_db.insert_user('user1', 'pass1')
        >>> user_db.delete_user_by_username('user1')
        """


    def validate_user_login(self, username, password):
        """
        Determine whether the user can log in, that is, the user is in the database and the password is correct
        :param username:str, the username of the user to validate.
        :param password:str, the password of the user to validate.
        :return:bool, representing whether the user can log in correctly
        >>> user_db = UserLoginDB("user_database.db")
        >>> user_db.create_table()
        >>> user_db.insert_user('user1', 'pass1')
        >>> user_db.validate_user_login('user1', 'pass1')
        True
        """
'''

import sqlite3


class UserLoginDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    @inspect_code
    def insert_user(self, username, password):
        self.cursor.execute('''
            INSERT INTO users (username, password)
            VALUES (?, ?)
        ''', (username, password))
        self.connection.commit()

    @inspect_code
    def search_user_by_username(self, username):
        self.cursor.execute('''
            SELECT * FROM users WHERE username = ?
        ''', (username,))
        user = self.cursor.fetchone()
        return user

    @inspect_code
    def delete_user_by_username(self, username):
        self.cursor.execute('''
            DELETE FROM users WHERE username = ?
        ''', (username,))
        self.connection.commit()

    @inspect_code
    def validate_user_login(self, username, password):
        user = self.search_user_by_username(username)
        if user is not None and user[1] == password:
            return True
        return False


import unittest
import os
from tempfile import gettempdir


class UserLoginDBTestInsertUser(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(gettempdir(), 'test_db.db')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        create_table_query = """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT,
                    password TEXT
                )
                """
        cursor.execute(create_table_query)

        conn.commit()
        conn.close()
        self.db = UserLoginDB(self.db_path)

    def tearDown(self):
        self.db.connection.close()
        os.unlink(self.db_path)

    def test_insert_user_1(self):
        self.db.insert_user('user1', 'pass1')
        user = self.db.search_user_by_username('user1')
        self.assertIsNotNone(user)
        self.assertEqual(user[0], 'user1')
        self.assertEqual(user[1], 'pass1')

    def test_insert_user_2(self):
        self.db.insert_user('user2', 'pass2')
        user = self.db.search_user_by_username('user2')
        self.assertIsNotNone(user)
        self.assertEqual(user[0], 'user2')
        self.assertEqual(user[1], 'pass2')

    def test_insert_user_3(self):
        self.db.insert_user('user3', 'pass3')
        user = self.db.search_user_by_username('user3')
        self.assertIsNotNone(user)
        self.assertEqual(user[0], 'user3')
        self.assertEqual(user[1], 'pass3')

    def test_insert_user_4(self):
        self.db.insert_user('user4', 'pass4')
        user = self.db.search_user_by_username('user4')
        self.assertIsNotNone(user)
        self.assertEqual(user[0], 'user4')
        self.assertEqual(user[1], 'pass4')

    def test_insert_user_5(self):
        self.db.insert_user('user5', 'pass5')
        user = self.db.search_user_by_username('user5')
        self.assertIsNotNone(user)
        self.assertEqual(user[0], 'user5')
        self.assertEqual(user[1], 'pass5')


class UserLoginDBTestSearchUserByUsername(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(gettempdir(), 'test_db.db')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        create_table_query = """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT,
                    password TEXT
                )
                """
        cursor.execute(create_table_query)

        conn.commit()
        conn.close()
        self.db = UserLoginDB(self.db_path)

    def tearDown(self):
        self.db.connection.close()
        os.unlink(self.db_path)

    def test_search_user_by_username_1(self):
        self.db.insert_user('user1', 'pass1')
        user = self.db.search_user_by_username('user1')
        self.assertIsNotNone(user)
        self.assertEqual(user[0], 'user1')
        self.assertEqual(user[1], 'pass1')

    def test_search_user_by_username_2(self):
        self.db.insert_user('user2', 'pass2')
        user = self.db.search_user_by_username('user2')
        self.assertIsNotNone(user)
        self.assertEqual(user[0], 'user2')
        self.assertEqual(user[1], 'pass2')

    def test_search_user_by_username_3(self):
        self.db.insert_user('user3', 'pass3')
        user = self.db.search_user_by_username('user3')
        self.assertIsNotNone(user)
        self.assertEqual(user[0], 'user3')
        self.assertEqual(user[1], 'pass3')

    def test_search_user_by_username_4(self):
        self.db.insert_user('user4', 'pass4')
        user = self.db.search_user_by_username('user4')
        self.assertIsNotNone(user)
        self.assertEqual(user[0], 'user4')
        self.assertEqual(user[1], 'pass4')

    def test_search_user_by_username_5(self):
        self.db.insert_user('user5', 'pass5')
        user = self.db.search_user_by_username('user5')
        self.assertIsNotNone(user)
        self.assertEqual(user[0], 'user5')
        self.assertEqual(user[1], 'pass5')


class UserLoginDBTestDeleteUserByUsername(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(gettempdir(), 'test_db.db')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        create_table_query = """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT,
                    password TEXT
                )
                """
        cursor.execute(create_table_query)

        conn.commit()
        conn.close()
        self.db = UserLoginDB(self.db_path)

    def tearDown(self):
        self.db.connection.close()
        os.unlink(self.db_path)

    def test_delete_user_by_username_1(self):
        self.db.insert_user('user1', 'pass1')
        self.db.delete_user_by_username('user1')
        user = self.db.search_user_by_username('user1')
        self.assertIsNone(user)

    def test_delete_user_by_username_2(self):
        self.db.insert_user('user2', 'pass2')
        self.db.delete_user_by_username('user2')
        user = self.db.search_user_by_username('user2')
        self.assertIsNone(user)

    def test_delete_user_by_username_3(self):
        self.db.insert_user('user3', 'pass3')
        self.db.delete_user_by_username('user3')
        user = self.db.search_user_by_username('user3')
        self.assertIsNone(user)

    def test_delete_user_by_username_4(self):
        self.db.insert_user('user4', 'pass4')
        self.db.delete_user_by_username('user4')
        user = self.db.search_user_by_username('user4')
        self.assertIsNone(user)

    def test_delete_user_by_username_5(self):
        self.db.insert_user('user5', 'pass5')
        self.db.delete_user_by_username('user5')
        user = self.db.search_user_by_username('user5')
        self.assertIsNone(user)


class UserLoginDBTestValidateUserLogin(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(gettempdir(), 'test_db.db')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        create_table_query = """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT,
                    password TEXT
                )
                """
        cursor.execute(create_table_query)

        conn.commit()
        conn.close()
        self.db = UserLoginDB(self.db_path)

    def tearDown(self):
        self.db.connection.close()
        os.unlink(self.db_path)

    def test_validate_user_login_1(self):
        self.db.insert_user('user1', 'pass1')
        valid = self.db.validate_user_login('user1', 'pass1')
        self.assertTrue(valid)

    def test_validate_user_login_2(self):
        self.db.insert_user('user1', 'pass1')
        invalid = self.db.validate_user_login('user1', 'wrongpass')
        self.assertFalse(invalid)

    def test_validate_user_login_3(self):
        valid = self.db.validate_user_login('nonexistentuser', 'somepass')
        self.assertFalse(valid)

    def test_validate_user_login_4(self):
        self.db.insert_user('user2', 'pass2')
        valid = self.db.validate_user_login('user2', 'pass2')
        self.assertTrue(valid)

    def test_validate_user_login_5(self):
        self.db.insert_user('user3', 'pass3')
        valid = self.db.validate_user_login('user3', 'pass3')
        self.assertTrue(valid)


class UserLoginDBTest(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(gettempdir(), 'test_db.db')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        create_table_query = """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT,
                    password TEXT
                )
                """
        cursor.execute(create_table_query)

        conn.commit()
        conn.close()
        self.db = UserLoginDB(self.db_path)

    def tearDown(self):
        self.db.connection.close()
        os.unlink(self.db_path)

    def test_UserLoginDB(self):
        self.db.insert_user('user1', 'pass1')
        user = self.db.search_user_by_username('user1')
        self.assertIsNotNone(user)
        self.assertEqual(user[0], 'user1')
        self.assertEqual(user[1], 'pass1')
        self.db.delete_user_by_username('user1')
        user = self.db.search_user_by_username('user1')
        self.assertIsNone(user)
        self.db.insert_user('user1', 'pass1')
        valid = self.db.validate_user_login('user1', 'pass1')
        self.assertTrue(valid)

