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
       jsonl_path = json_base + "/BookManagement.jsonl"
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
# This is a class as managing books system, which supports to add and remove books from the inventory dict, view the inventory, and check the quantity of a specific book.

class BookManagement:
    def __init__(self):
        """
        Initialize the inventory of Book Manager.
        """
        self.inventory = {}

    def add_book(self, title, quantity=1):
        """
        Add one or several books to inventory which is sorted by book title.
        :param title: str, the book title
        :param quantity: int, default value is 1.
        """

    def remove_book(self, title, quantity):
        """
        Remove one or several books from inventory which is sorted by book title.
        Raise false while get invalid input.
        :param title: str, the book title
        :param quantity: int
        """

    def view_inventory(self):
        """
        Get the inventory of the Book Management.
        :return self.inventory: dictionary, {title(str): quantity(int), ...}
        >>> bookManagement = BookManagement()
        >>> bookManagement.add_book("book1", 1)
        >>> bookManagement.add_book("book2", 1)
        >>> bookManagement.view_inventory()
        {'book1': 1, 'book2': 1}
        """

    def view_book_quantity(self, title):
        """
        Get the quantity of a book.
        :param title: str, the title of the book.
        :return quantity: the quantity of this book title. return 0 when the title does not exist in self.invenroty
        >>> bookManagement = BookManagement()
        >>> bookManagement.add_book("book1", 1)
        >>> bookManagement.view_book_quantity("book3")
        0
        """
'''


class BookManagement:
    def __init__(self):
        self.inventory = {}

    @inspect_code
    def add_book(self, title, quantity=1):
        if title in self.inventory:
            self.inventory[title] += quantity
        else:
            self.inventory[title] = quantity

    @inspect_code
    def remove_book(self, title, quantity):
        if title not in self.inventory or self.inventory[title] < quantity:
            raise False
        self.inventory[title] -= quantity
        if self.inventory[title] == 0:
            del (self.inventory[title])

    @inspect_code
    def view_inventory(self):
        return self.inventory

    @inspect_code
    def view_book_quantity(self, title):
        if title not in self.inventory:
            return 0
        return self.inventory[title]



import unittest


class BookManagementTestAddBook(unittest.TestCase):
    def test_add_book_1(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1")
        self.assertEqual({"book1": 1}, bookManagement.inventory)

    def test_add_book_2(self):
        bookManagement = BookManagement()
        self.assertEqual({}, bookManagement.inventory)

    def test_add_book_3(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1")
        bookManagement.add_book("book1", 2)
        self.assertEqual({"book1": 3}, bookManagement.inventory)

    def test_add_book_4(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1", 2)
        self.assertEqual({"book1": 2}, bookManagement.inventory)

    def test_add_book_5(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1", 2)
        bookManagement.add_book("book1")
        self.assertEqual({"book1": 3}, bookManagement.inventory)


class BookManagementTestRemoveBook(unittest.TestCase):
    def setUp(self) -> None:
        self.bookManagement = BookManagement()
        self.bookManagement.add_book("book1", 2)
        self.bookManagement.add_book("book2")

    # remove all this title books
    def test_remove_book_1(self):
        self.bookManagement.remove_book("book1", 2)
        self.assertEqual(self.bookManagement.inventory, {"book2": 1})

    # remove part
    def test_remove_book_2(self):
        self.bookManagement.remove_book("book1", 1)
        self.assertEqual(self.bookManagement.inventory, {"book1": 1, "book2": 1})

    # remove the title that doesn't exist
    def test_remove_book_3(self):
        with self.assertRaises(Exception):
            self.bookManagement.remove_book("book3", 1)

    # invalid quantity
    def test_remove_book_4(self):
        with self.assertRaises(Exception):
            self.bookManagement.remove_book("book2", 2)

    def test_remove_book_5(self):
        with self.assertRaises(Exception):
            self.bookManagement.remove_book("book2", 5)


class BookManagementTestViewInventory(unittest.TestCase):
    def test_view_inventory_1(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1", 2)
        bookManagement.add_book("book2")
        expected = {"book1": 2, "book2": 1}
        self.assertEqual(expected, bookManagement.inventory)

    def test_view_inventory_2(self):
        bookManagement = BookManagement()
        expected = {}
        self.assertEqual(expected, bookManagement.inventory)

    def test_view_inventory_3(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1", 2)
        bookManagement.add_book("book2")
        expected = {"book1": 2, "book2": 1}
        self.assertEqual(expected, bookManagement.inventory)

    def test_view_inventory_4(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1", 2)
        bookManagement.add_book("book2")
        bookManagement.remove_book("book1", 2)
        expected = {"book2": 1}
        self.assertEqual(expected, bookManagement.inventory)

    def test_view_inventory_5(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1", 2)
        bookManagement.add_book("book2", 1)
        bookManagement.remove_book("book1", 2)
        bookManagement.remove_book("book2",1)
        expected = {}
        self.assertEqual(expected, bookManagement.inventory)


class BookManagementTestViewBookQuantity(unittest.TestCase):
    def test_view_book_quantity_1(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1", 2)
        self.assertEqual(2, bookManagement.view_book_quantity("book1"))

    def test_view_book_quantity_2(self):
        bookManagement = BookManagement()
        self.assertEqual(0, bookManagement.view_book_quantity("book1"))

    def test_view_book_quantity_3(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1", 2)
        self.assertEqual(2, bookManagement.view_book_quantity("book1"))

    def test_view_book_quantity_4(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1", 2)
        bookManagement.remove_book("book1", 2)
        self.assertEqual(0, bookManagement.view_book_quantity("book1"))

    def test_view_book_quantity_5(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1", 3)
        bookManagement.remove_book("book1", 2)
        self.assertEqual(1, bookManagement.view_book_quantity("book1"))


class BookManagementTestMain(unittest.TestCase):
    def test_main(self):
        bookManagement = BookManagement()
        bookManagement.add_book("book1", 2)
        bookManagement.add_book("book2")
        self.assertEqual(bookManagement.view_inventory(), {"book1": 2, "book2": 1})

        bookManagement.remove_book("book2", 1)
        self.assertEqual(bookManagement.view_inventory(), {"book1": 2})
        self.assertEqual(0, bookManagement.view_book_quantity("book2"))