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
       jsonl_path = json_base + "/MovieTicketDB.jsonl"
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
# This is a class for movie database operations, which allows for inserting movie information, searching for movie information by name, and deleting movie information by name.

import sqlite3

class MovieTicketDB:
    def __init__(self, db_name):
        """
        Initializes the MovieTicketDB object with the specified database name.
        :param db_name: str, the name of the SQLite database.
        """
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()


    def create_table(self):
        """
        Creates a "tickets" table in the database if it does not exist already.Fields include ID of type int, movie name of type str, theater name of type str, seat number of type str, and customer name of type str
        :return: None
        """

    def insert_ticket(self, movie_name, theater_name, seat_number, customer_name):
        """
        Inserts a new ticket into the "tickets" table.
        :param movie_name: str, the name of the movie.
        :param theater_name: str, the name of the theater.
        :param seat_number: str, the seat number.
        :param customer_name: str, the name of the customer.
        :return: None
        """

    def search_tickets_by_customer(self, customer_name):
        """
        Searches for tickets in the "tickets" table by customer name.
        :param customer_name: str, the name of the customer to search for.
        :return: list of tuples, the rows from the "tickets" table that match the search criteria.
        >>> ticket_db = MovieTicketDB("ticket_database.db")
        >>> ticket_db.create_table()
        >>> ticket_db.insert_ticket("Movie A", "Theater 1", "A1", "John Doe")
        >>> result = ticket_db.search_tickets_by_customer("John Doe")
        len(result) = 1
        """

    def delete_ticket(self, ticket_id):
        """
        Deletes a ticket from the "tickets" table by ticket ID.
        :param ticket_id: int, the ID of the ticket to delete.
        :return: None
        """

'''

import sqlite3


class MovieTicketDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    @inspect_code
    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY,
                movie_name TEXT,
                theater_name TEXT,
                seat_number TEXT,
                customer_name TEXT
            )
        ''')
        self.connection.commit()

    @inspect_code
    def insert_ticket(self, movie_name, theater_name, seat_number, customer_name):
        self.cursor.execute('''
            INSERT INTO tickets (movie_name, theater_name, seat_number, customer_name)
            VALUES (?, ?, ?, ?)
        ''', (movie_name, theater_name, seat_number, customer_name))
        self.connection.commit()

    @inspect_code
    def search_tickets_by_customer(self, customer_name):
        self.cursor.execute('''
            SELECT * FROM tickets WHERE customer_name = ?
        ''', (customer_name,))
        tickets = self.cursor.fetchall()
        return tickets

    @inspect_code
    def delete_ticket(self, ticket_id):
        self.cursor.execute('''
            DELETE FROM tickets WHERE id = ?
        ''', (ticket_id,))
        self.connection.commit()



import unittest
import os


class MovieTicketDBTestInsertTicket(unittest.TestCase):
    def setUp(self):
        self.db_name = 'test_database.db'
        self.db = MovieTicketDB(self.db_name)

    def tearDown(self):
        self.db.connection.close()
        os.remove(self.db_name)

    def test_insert_ticket_1(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'John Doe')
        tickets = self.db.search_tickets_by_customer('John Doe')
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket[1], 'Avengers: Endgame')
        self.assertEqual(ticket[2], 'Cinema 1')
        self.assertEqual(ticket[3], 'A1')
        self.assertEqual(ticket[4], 'John Doe')

    def test_insert_ticket_2(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'aaa')
        tickets = self.db.search_tickets_by_customer('aaa')
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket[1], 'Avengers: Endgame')
        self.assertEqual(ticket[2], 'Cinema 1')
        self.assertEqual(ticket[3], 'A1')
        self.assertEqual(ticket[4], 'aaa')

    def test_insert_ticket_3(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'bbb')
        tickets = self.db.search_tickets_by_customer('bbb')
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket[1], 'Avengers: Endgame')
        self.assertEqual(ticket[2], 'Cinema 1')
        self.assertEqual(ticket[3], 'A1')
        self.assertEqual(ticket[4], 'bbb')

    def test_insert_ticket_4(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'ccc')
        tickets = self.db.search_tickets_by_customer('ccc')
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket[1], 'Avengers: Endgame')
        self.assertEqual(ticket[2], 'Cinema 1')
        self.assertEqual(ticket[3], 'A1')
        self.assertEqual(ticket[4], 'ccc')

    def test_insert_ticket_5(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'ddd')
        tickets = self.db.search_tickets_by_customer('ddd')
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket[1], 'Avengers: Endgame')
        self.assertEqual(ticket[2], 'Cinema 1')
        self.assertEqual(ticket[3], 'A1')
        self.assertEqual(ticket[4], 'ddd')


class MovieTicketDBTestSearchTicketsByCustomer(unittest.TestCase):
    def setUp(self):
        self.db_name = 'test_database.db'
        self.db = MovieTicketDB(self.db_name)

    def tearDown(self):
        self.db.connection.close()
        os.remove(self.db_name)

    def test_search_tickets_by_customer_1(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'John Doe')
        tickets = self.db.search_tickets_by_customer('John Doe')
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket[1], 'Avengers: Endgame')
        self.assertEqual(ticket[2], 'Cinema 1')
        self.assertEqual(ticket[3], 'A1')
        self.assertEqual(ticket[4], 'John Doe')

    def test_search_tickets_by_customer_2(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'aaa')
        tickets = self.db.search_tickets_by_customer('aaa')
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket[1], 'Avengers: Endgame')
        self.assertEqual(ticket[2], 'Cinema 1')
        self.assertEqual(ticket[3], 'A1')
        self.assertEqual(ticket[4], 'aaa')

    def test_search_tickets_by_customer_3(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'bbb')
        tickets = self.db.search_tickets_by_customer('bbb')
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket[1], 'Avengers: Endgame')
        self.assertEqual(ticket[2], 'Cinema 1')
        self.assertEqual(ticket[3], 'A1')
        self.assertEqual(ticket[4], 'bbb')

    def test_search_tickets_by_customer_4(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'ccc')
        tickets = self.db.search_tickets_by_customer('ccc')
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket[1], 'Avengers: Endgame')
        self.assertEqual(ticket[2], 'Cinema 1')
        self.assertEqual(ticket[3], 'A1')
        self.assertEqual(ticket[4], 'ccc')

    def test_search_tickets_by_customer_5(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'ddd')
        tickets = self.db.search_tickets_by_customer('ddd')
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket[1], 'Avengers: Endgame')
        self.assertEqual(ticket[2], 'Cinema 1')
        self.assertEqual(ticket[3], 'A1')
        self.assertEqual(ticket[4], 'ddd')


class MovieTicketDBTestDeleteTicket(unittest.TestCase):
    def setUp(self):
        self.db_name = 'test_database.db'
        self.db = MovieTicketDB(self.db_name)

    def tearDown(self):
        self.db.connection.close()
        os.remove(self.db_name)

    def test_delete_ticket_1(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'John Doe')
        tickets = self.db.search_tickets_by_customer('John Doe')
        self.assertEqual(len(tickets), 1)
        ticket_id = tickets[0][0]
        self.db.delete_ticket(ticket_id)
        tickets = self.db.search_tickets_by_customer('John Doe')
        self.assertEqual(len(tickets), 0)

    def test_delete_ticket_2(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'aaa')
        tickets = self.db.search_tickets_by_customer('aaa')
        self.assertEqual(len(tickets), 1)
        ticket_id = tickets[0][0]
        self.db.delete_ticket(ticket_id)
        tickets = self.db.search_tickets_by_customer('aaa')
        self.assertEqual(len(tickets), 0)

    def test_delete_ticket_3(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'bbb')
        tickets = self.db.search_tickets_by_customer('bbb')
        self.assertEqual(len(tickets), 1)
        ticket_id = tickets[0][0]
        self.db.delete_ticket(ticket_id)
        tickets = self.db.search_tickets_by_customer('bbb')
        self.assertEqual(len(tickets), 0)

    def test_delete_ticket_4(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'ccc')
        tickets = self.db.search_tickets_by_customer('ccc')
        self.assertEqual(len(tickets), 1)
        ticket_id = tickets[0][0]
        self.db.delete_ticket(ticket_id)
        tickets = self.db.search_tickets_by_customer('ccc')
        self.assertEqual(len(tickets), 0)

    def test_delete_ticket_5(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'ddd')
        tickets = self.db.search_tickets_by_customer('ddd')
        self.assertEqual(len(tickets), 1)
        ticket_id = tickets[0][0]
        self.db.delete_ticket(ticket_id)
        tickets = self.db.search_tickets_by_customer('ddd')
        self.assertEqual(len(tickets), 0)


class MovieTicketDBTest(unittest.TestCase):
    def setUp(self):
        self.db_name = 'test_database.db'
        self.db = MovieTicketDB(self.db_name)

    def tearDown(self):
        self.db.connection.close()
        os.remove(self.db_name)

    def test_MovieTicketDB(self):
        self.db.insert_ticket('Avengers: Endgame', 'Cinema 1', 'A1', 'John Doe')
        tickets = self.db.search_tickets_by_customer('John Doe')
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket[1], 'Avengers: Endgame')
        self.assertEqual(ticket[2], 'Cinema 1')
        self.assertEqual(ticket[3], 'A1')
        self.assertEqual(ticket[4], 'John Doe')
        ticket_id = tickets[0][0]
        self.db.delete_ticket(ticket_id)
        tickets = self.db.search_tickets_by_customer('John Doe')
        self.assertEqual(len(tickets), 0)
