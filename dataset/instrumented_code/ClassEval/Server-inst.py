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
       jsonl_path = json_base + "/Server.jsonl"
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
# This is a class as a server, which handles a white list, message sending and receiving, and information display.

class Server:

    def __init__(self):
        """
        Initialize the whitelist as an empty list, and initialize the sending and receiving information as an empty dictionary
        """
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}



    def add_white_list(self, addr):
        """
        Add an address to the whitelist and do nothing if it already exists
        :param addr: int, address to be added
        :return: new whitelist, return False if the address already exists
        >>> server = Server()
        >>> server.add_white_list(88)
        [88]
        """

    def del_white_list(self, addr):
        """
        Remove an address from the whitelist and do nothing if it does not exist
        :param addr: int, address to be deleted
        :return: new whitelist, return False if the address does not exist
        >>> server.add_white_list(88)
        >>> server.del_white_list(88)
        []
        """

    def recv(self, info):
        """
        Receive information containing address and content. If the address is on the whitelist, receive the content; otherwise, do not receive it
        :param info: dict, information dictionary containing address and content
        :return: if successfully received, return the content of the infomation; otherwise, return False
        >>> server.recv({"addr":88,"content":"abc"})
        abc
        """

    def send(self, info):
        """
        Send information containing address and content
        :param info: dict, information dictionary containing address and content
        :return: if successfully sent, return nothing; otherwise, return a string indicating an error message
        >>> server.send({"addr":66,"content":"ABC"})
        self.send_struct = {"addr":66,"content":"ABC"}
        """

    def show(self, type):
        """
        Returns struct of the specified type
        :param type: string, the type of struct to be returned, which can be 'send' or 'receive'
        :return: if type is equal to 'send' or 'receive', return the corresponding struct; otherwise, return False
        >>> server.recv({"addr":88,"content":"abc"})
        >>> server.send({"addr":66,"content":"ABC"})
        >>> server.show("send")
        {"addr":66,"content":"ABC"}
        """

'''


class Server:

    def __init__(self):
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    @inspect_code
    def add_white_list(self, addr):
        if addr in self.white_list:
            return False
        else:
            self.white_list.append(addr)
            return self.white_list

    @inspect_code
    def del_white_list(self, addr):
        if addr not in self.white_list:
            return False
        else:
            self.white_list.remove(addr)
            return self.white_list

    @inspect_code
    def recv(self, info):
        if not isinstance(info, dict) or "addr" not in info or "content" not in info:
            return -1
        addr = info["addr"]
        content = info["content"]
        if addr not in self.white_list:
            return False
        else:
            self.receive_struct = {"addr": addr, "content": content}
            return self.receive_struct["content"]

    @inspect_code
    def send(self, info):
        if not isinstance(info, dict) or "addr" not in info or "content" not in info:
            return "info structure is not correct"
        self.send_struct = {"addr": info["addr"], "content": info["content"]}

    @inspect_code
    def show(self, type):
        if type == "send":
            return self.send_struct
        elif type == "receive":
            return self.receive_struct
        else:
            return False

import unittest


class ServerTestAddWhiteList(unittest.TestCase):
    def test_add_white_list_1(self):
        server = Server()
        server.add_white_list(88)
        self.assertEqual(server.white_list, [88])

    def test_add_white_list_2(self):
        server = Server()
        server.add_white_list(88)
        self.assertEqual(server.add_white_list(88), False)

    def test_add_white_list_3(self):
        server = Server()
        server.add_white_list(88)
        server.add_white_list(11)
        self.assertEqual(server.add_white_list(11), False)

    def test_add_white_list_4(self):
        server = Server()
        server.add_white_list(11)
        self.assertEqual(server.white_list, [11])

    def test_add_white_list_5(self):
        server = Server()
        server.add_white_list(88)
        server.add_white_list(11)
        server.add_white_list(22)
        self.assertEqual(server.add_white_list(22), False)


class ServerTestDelWhiteList(unittest.TestCase):
    def test_del_white_list_1(self):
        server = Server()
        server.add_white_list(88)
        server.del_white_list(88)
        self.assertEqual(server.white_list, [])

    def test_del_white_list_2(self):
        server = Server()
        self.assertEqual(server.del_white_list(88), False)

    def test_del_white_list_3(self):
        server = Server()
        self.assertEqual(server.del_white_list(11), False)

    def test_del_white_list_4(self):
        server = Server()
        self.assertEqual(server.del_white_list(22), False)

    def test_del_white_list_5(self):
        server = Server()
        server.add_white_list(11)
        self.assertEqual(server.del_white_list(22), False)


class ServerTestRecv(unittest.TestCase):
    def test_recv_1(self):
        server = Server()
        server.add_white_list(88)
        server.recv({"addr": 88, "content": "abc"})
        self.assertEqual(server.receive_struct, {"addr": 88, "content": "abc"})

    def test_recv_2(self):
        server = Server()
        server.add_white_list(88)
        flag = server.recv({"addr": 66, "content": "abc"})
        self.assertEqual(server.receive_struct, {})
        self.assertEqual(flag, False)

    def test_recv_3(self):
        server = Server()
        flag = server.recv([88])
        self.assertEqual(server.receive_struct, {})
        self.assertEqual(flag, -1)

    def test_recv_4(self):
        server = Server()
        flag = server.recv({"addr": 88})
        self.assertEqual(server.receive_struct, {})
        self.assertEqual(flag, -1)

    def test_recv_5(self):
        server = Server()
        flag = server.recv({"content": "abc"})
        self.assertEqual(server.receive_struct, {})
        self.assertEqual(flag, -1)


class ServerTestSend(unittest.TestCase):
    def test_send_1(self):
        server = Server()
        server.send({"addr": 88, "content": "abc"})
        self.assertEqual(server.send_struct, {"addr": 88, "content": "abc"})

    def test_send_2(self):
        server = Server()
        flag = server.send({"addr": 88})
        self.assertEqual(flag, "info structure is not correct")

    def test_send_3(self):
        server = Server()
        flag = server.send({"content": "abc"})
        self.assertEqual(flag, "info structure is not correct")

    def test_send_4(self):
        server = Server()
        flag = server.send([])
        self.assertEqual(flag, "info structure is not correct")

    def test_send_5(self):
        server = Server()
        server.send({"addr": 66, "content": "abc"})
        self.assertEqual(server.send_struct, {"addr": 66, "content": "abc"})


class ServerTestShow(unittest.TestCase):
    def test_show_1(self):
        server = Server()
        server.add_white_list(66)
        server.send({"addr": 88, "content": "abc"})
        server.recv({"addr": 66, "content": "ABC"})
        self.assertEqual(server.show("send"), {"addr": 88, "content": "abc"})

    def test_show_2(self):
        server = Server()
        server.add_white_list(66)
        server.send({"addr": 88, "content": "abc"})
        server.recv({"addr": 66, "content": "ABC"})
        self.assertEqual(server.show("receive"), {"addr": 66, "content": "ABC"})

    def test_show_3(self):
        server = Server()
        server.add_white_list(66)
        server.send({"addr": 88, "content": "abc"})
        server.recv({"addr": 66, "content": "ABC"})
        self.assertEqual(server.show("abcdefg"), False)

    def test_show_4(self):
        server = Server()
        server.add_white_list(66)
        server.send({"addr": 11, "content": "abc"})
        server.recv({"addr": 66, "content": "ABC"})
        self.assertEqual(server.show("send"), {"addr": 11, "content": "abc"})

    def test_show_5(self):
        server = Server()
        server.add_white_list(66)
        server.send({"addr": 22, "content": "abc"})
        server.recv({"addr": 66, "content": "ABC"})
        self.assertEqual(server.show("send"), {"addr": 22, "content": "abc"})


class ServerTest(unittest.TestCase):
    def test_server(self):
        server = Server()
        server.add_white_list(88)
        self.assertEqual(server.white_list, [88])
        server.del_white_list(88)
        self.assertEqual(server.white_list, [])
        server.add_white_list(88)
        server.recv({"addr": 88, "content": "abc"})
        self.assertEqual(server.receive_struct, {"addr": 88, "content": "abc"})
        server.send({"addr": 66, "content": "ABC"})
        self.assertEqual(server.send_struct, {"addr": 66, "content": "ABC"})
        server.recv({"addr": 88, "content": "abc"})
        self.assertEqual(server.show("receive"), {"addr": 88, "content": "abc"})
