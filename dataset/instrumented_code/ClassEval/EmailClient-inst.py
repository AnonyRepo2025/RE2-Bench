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
       jsonl_path = json_base + "/EmailClient.jsonl"
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
# This is a class that serves as an email client, implementing functions such as checking emails, determining whether there is sufficient space, and cleaning up space

from datetime import datetime

class EmailClient:
    def __init__(self, addr, capacity) -> None:
        """
        Initializes the EmailClient class with the email address and the capacity of the email box.
        :param addr: The email address, str.
        :param capacity: The capacity of the email box, float.
        """
        self.addr = addr
        self.capacity = capacity
        self.inbox = []

    def send_to(self, recv, content, size):
        """
        Sends an email to the given email address.
        :param recv: The email address of the receiver, str.
        :param content: The content of the email, str.
        :param size: The size of the email, float.
        :return: True if the email is sent successfully, False if the receiver's email box is full.
        >>> sender = EmailClient('sender@example.com', 100)
        >>> receiver = EmailClient('receiver@example.com', 50)
        >>> sender.send_to(receiver, 'Hello', 10)
        True
        >>> receiver.inbox
        {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10, 'time': '2023-07-13 11:36:40', 'state': 'unread'}

        """

    def fetch(self):
        """
        Retrieves the first unread email in the email box and marks it as read.
        :return: The first unread email in the email box, dict.
        >>> sender = EmailClient('sender@example.com', 100)
        >>> receiver = EmailClient('receiver@example.com', 50)
        >>> receiver.inbox = [{'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10, 'time': '2023-07-13 11:36:40', 'state': 'unread'}]
        >>> receiver.fetch()
        {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10, 'time': '2023-07-13 11:36:40', 'state': 'read'}

        """

    def is_full_with_one_more_email(self, size):
        """
        Determines whether the email box is full after adding an email of the given size.
        :param size: The size of the email, float.
        :return: True if the email box is full, False otherwise.
        >>> sender = EmailClient('sender@example.com', 100)
        >>> receiver = EmailClient('receiver@example.com', 50)
        >>> receiver.is_full_with_one_more_email(10)
        False

        """

    def get_occupied_size(self):
        """
        Gets the total size of the emails in the email box.
        :return: The total size of the emails in the email box, float.
        >>> sender = EmailClient('sender@example.com', 100)
        >>> receiver = EmailClient('receiver@example.com', 50)
        >>> sender.inbox = [{'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10, 'time': datetime.now, 'state': 'unread'}]
        >>> sender.get_occupied_size()
        10

        """

    def clear_inbox(self, size):
        """
        Clears the email box by deleting the oldest emails until the email box has enough space to accommodate the given size.
        :param size: The size of the email, float.
        >>> sender = EmailClient('sender@example.com', 100)
        >>> receiver = EmailClient('receiver@example.com', 50)
        >>> receiver.inbox = [{'size': 10},{'size': 20},{'size': 15}]
        >>> receiver.clear_inbox(30)
        >>> receiver.inbox
        [{'size': 15}]

        """
'''

from datetime import datetime

class EmailClient:
    def __init__(self, addr, capacity) -> None:
        self.addr = addr
        self.capacity = capacity
        self.inbox = []
    
    @inspect_code
    def send_to(self, recv, content, size):
        if not recv.is_full_with_one_more_email(size):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            email = {
                "sender": self.addr,
                "receiver": recv.addr,
                "content": content,
                "size": size,
                "time": timestamp,
                "state": "unread"
            }
            recv.inbox.append(email)
            return True
        else:
            self.clear_inbox(size)
            return False
    
    @inspect_code
    def fetch(self):
        if len(self.inbox) == 0:
            return None
        for i in range(len(self.inbox)):
            if self.inbox[i]['state'] == "unread":
                self.inbox[i]['state'] = "read"
                return self.inbox[i]
        return None

    @inspect_code
    def is_full_with_one_more_email(self, size):
        occupied_size = self.get_occupied_size()
        return True if occupied_size + size > self.capacity else False
        
    @inspect_code
    def get_occupied_size(self):
        occupied_size = 0
        for email in self.inbox:
            occupied_size += email["size"]
        return occupied_size

    @inspect_code
    def clear_inbox(self, size):
        if len(self.addr) == 0:
            return
        freed_space = 0
        while freed_space < size and self.inbox:
            email = self.inbox[0]
            freed_space += email['size']
            del self.inbox[0]

import unittest

class EmailClientTestSendTo(unittest.TestCase):
    def test_send_to(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.assertTrue(sender.send_to(receiver, 'Hello', 10))
        self.assertEqual(receiver.inbox[0], {"sender": 'sender@example.com','receiver': 'receiver@example.com','content': 'Hello','size': 10,'time': timestamp,'state': 'unread'})

    def test_send_to_2(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 0)
        self.assertFalse(sender.send_to(receiver, 'Hello', 10))

    def test_send_to_3(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        receiver.inbox = [{'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 50, 'time': '2021-01-01 00:00:00', 'state': 'unread'}]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.assertFalse(sender.send_to(receiver, 'Hello', 10))
        self.assertEqual(receiver.inbox, [{'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 50, 'time': '2021-01-01 00:00:00', 'state': 'unread'}])

    def test_send_to_4(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 30)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.assertTrue(sender.send_to(receiver, 'Hello', 20))
        self.assertEqual(receiver.inbox, [{'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 20, 'time': timestamp, 'state': 'unread'}])

    def test_send_to_5(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 30)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.assertTrue(sender.send_to(receiver, 'bye', 20))
        self.assertEqual(receiver.inbox, [{'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'bye', 'size': 20, 'time': timestamp, 'state': 'unread'}])
class EmailClientTestFetch(unittest.TestCase):
    def test_fetch(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        receiver.inbox = [
            {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10,
             'time': timestamp, 'state': 'unread'}]
        self.assertEqual(receiver.fetch(), {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10, 'time':timestamp, 'state': 'read'})

    def test_fetch_2(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.assertEqual(receiver.fetch(),None)

    def test_fetch_3(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        receiver.inbox = [
            {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10,
             'time': timestamp, 'state': 'read'}]
        self.assertEqual(receiver.fetch(), None)

    def test_fetch_4(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        receiver.inbox = [
            {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10,
             'time':  '2021-01-01 00:00:00', 'state': 'unread'},
            {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10,
             'time': timestamp, 'state': 'unread'}]
        self.assertEqual(receiver.fetch(), {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10,
             'time':  '2021-01-01 00:00:00', 'state': 'read'})

    def test_fetch_5(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        receiver.inbox = [
            {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10,
             'time': '2021-01-01 00:00:00', 'state': 'read'},
            {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10,
             'time': timestamp, 'state': 'unread'}]
        self.assertEqual(receiver.fetch(), {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10,
             'time': timestamp, 'state': 'read'})

class EmailClientTestIsFullWithOneMoreEmail(unittest.TestCase):
    def test_is_full_with_one_more_email(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        self.assertFalse(receiver.is_full_with_one_more_email(10))

    def test_is_full_with_one_more_email_2(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 0)
        self.assertTrue(receiver.is_full_with_one_more_email(10))

    def test_is_full_with_one_more_email_3(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 10)
        self.assertFalse(receiver.is_full_with_one_more_email(10))

    def test_is_full_with_one_more_email_4(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 10)
        self.assertTrue(receiver.is_full_with_one_more_email(20))

    def test_is_full_with_one_more_email_5(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 20)
        self.assertFalse(receiver.is_full_with_one_more_email(20))

class EmailClientTestGetOccupiedSize(unittest.TestCase):
    def test_get_occupied_size(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        sender.inbox = [{'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10, 'time': datetime.now, 'state': 'unread'}]
        self.assertEqual(sender.get_occupied_size(), 10)

    def test_get_occupied_size_2(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        sender.inbox =[]
        self.assertEqual(sender.get_occupied_size(), 0)

    def test_get_occupied_size_3(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        sender.inbox = [
            {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 20,
             'time': datetime.now, 'state': 'unread'}]
        self.assertEqual(sender.get_occupied_size(), 20)

    def test_get_occupied_size_4(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        sender.inbox = [
            {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 20,
             'time': datetime.now, 'state': 'unread'},
            {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 30,
             'time': datetime.now, 'state': 'unread'}]
        self.assertEqual(sender.get_occupied_size(), 50)

    def test_get_occupied_size_5(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        sender.inbox = [
            {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 20,
             'time': datetime.now, 'state': 'unread'},
            {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 60,
             'time': datetime.now, 'state': 'unread'}]
        self.assertEqual(sender.get_occupied_size(), 80)

class EmailClientTestClearInbox(unittest.TestCase):
    def test_clear_inbox(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        receiver.inbox = [{'size': 10},{'size': 20},{'size': 15}]
        receiver.clear_inbox(30)
        self.assertEqual(receiver.inbox, [{'size': 15}])

    def test_clear_inbox_2(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('', 50)
        receiver.inbox = [{'size': 10},{'size': 20},{'size': 15}]
        self.assertEqual(receiver.clear_inbox(30),None)
        self.assertEqual(receiver.inbox, [{'size': 10},{'size': 20},{'size': 15}])

    def test_clear_inbox_3(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        receiver.inbox = [{'size': 10}, {'size': 20}, {'size': 15}]
        self.assertEqual(receiver.clear_inbox(50), None)

    def test_clear_inbox_4(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        receiver.inbox = [{'size': 10}, {'size': 20}, {'size': 15}]
        receiver.clear_inbox(45)
        self.assertEqual(receiver.inbox, [])
    def test_clear_inbox_5(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        receiver.inbox = [{'size': 10}, {'size': 20}, {'size': 15}]
        receiver.clear_inbox(10)
        self.assertEqual(receiver.inbox, [{'size': 20}, {'size': 15}])




class EmailClientTestMain(unittest.TestCase):
    def test_main(self):
        sender = EmailClient('sender@example.com', 100)
        receiver = EmailClient('receiver@example.com', 50)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.assertTrue(sender.send_to(receiver, 'Hello', 10))
        self.assertEqual(receiver.inbox[0], {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10, 'time': timestamp, 'state': 'unread'})
        self.assertEqual(receiver.fetch(), {'sender': 'sender@example.com', 'receiver': 'receiver@example.com', 'content': 'Hello', 'size': 10, 'time': timestamp, 'state': 'read'})
        self.assertFalse(receiver.is_full_with_one_more_email(10))
        self.assertEqual(receiver.get_occupied_size(), 10)
        receiver.inbox = [{'size': 10},{'size': 20},{'size': 15}]
        receiver.clear_inbox(30)
        self.assertEqual(receiver.inbox, [{'size': 15}])

