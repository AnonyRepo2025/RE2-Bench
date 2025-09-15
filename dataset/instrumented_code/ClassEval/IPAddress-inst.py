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
       jsonl_path = json_base + "/IPAddress.jsonl"
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
# This is a class to process IP Address, including validating, getting the octets and obtaining the binary representation of a valid IP address.

class IPAddress:
    def __init__(self, ip_address):
        """
        Initialize the IP address to the specified address
        :param ip_address:string
        """
        self.ip_address = ip_address


    def is_valid(self):
        """
        Judge whether the IP address is valid, that is, whether the IP address is composed of four Decimal digits separated by '.'. Each digit is greater than or equal to 0 and less than or equal to 255
        :return: bool
        >>> ipaddress = IPAddress("10.10.10.10")
        >>> ipaddress.is_valid()
        True
        """


    def get_octets(self):
        """
        If the IP address is valid, the list of four decimal numbers separated by "." constituting the IP address is returned; otherwise, an empty list is returned
        :return: list
        >>> ipaddress = IPAddress("10.10.10.10")
        >>> ipaddress.get_octets()
        ["10", "10", "10", "10"]
        """


    def get_binary(self):
        """
        If the IP address is valid, return the binary form of the IP address; otherwise, return ''
        :return: string
        >>> ipaddress = IPAddress("10.10.10.10")
        >>> ipaddress.get_binary()
        "00001010.00001010.00001010.00001010"
        """
'''


class IPAddress:
    def __init__(self, ip_address):
        self.ip_address = ip_address

    @inspect_code
    def is_valid(self):
        octets = self.ip_address.split('.')
        if len(octets) != 4:
            return False
        for octet in octets:
            if not octet.isdigit() or int(octet) < 0 or int(octet) > 255:
                return False
        return True

    @inspect_code
    def get_octets(self):
        if self.is_valid():
            return self.ip_address.split('.')
        else:
            return []

    @inspect_code
    def get_binary(self):
        if self.is_valid():
            binary_octets = []
            for octet in self.get_octets():
                binary_octets.append(format(int(octet), '08b'))
            return '.'.join(binary_octets)
        else:
            return ''

import unittest


class IPAddressTestIsValid(unittest.TestCase):
    def test_is_valid_1(self):
        ipaddress = IPAddress("10.10.10.10")
        self.assertEqual(ipaddress.is_valid(), True)

    def test_is_valid_2(self):
        ipaddress = IPAddress("-1.10.10.10")
        self.assertEqual(ipaddress.is_valid(), False)

    def test_is_valid_3(self):
        ipaddress = IPAddress("10.10.10")
        self.assertEqual(ipaddress.is_valid(), False)

    def test_is_valid_4(self):
        ipaddress = IPAddress("a.10.10.10")
        self.assertEqual(ipaddress.is_valid(), False)

    def test_is_valid_5(self):
        ipaddress = IPAddress("300.10.10.10")
        self.assertEqual(ipaddress.is_valid(), False)


class IPAddressTestGetOctets(unittest.TestCase):
    def test_get_octets_1(self):
        ipaddress = IPAddress("10.10.10.10")
        self.assertEqual(ipaddress.get_octets(), ["10", "10", "10", "10"])

    def test_get_octets_2(self):
        ipaddress = IPAddress("a.10.10.10")
        self.assertEqual(ipaddress.get_octets(), [])

    def test_get_octets_3(self):
        ipaddress = IPAddress("-1.10.10.10")
        self.assertEqual(ipaddress.get_octets(), [])

    def test_get_octets_4(self):
        ipaddress = IPAddress("300.10.10.10")
        self.assertEqual(ipaddress.get_octets(), [])

    def test_get_octets_5(self):
        ipaddress = IPAddress(".10.10.10")
        self.assertEqual(ipaddress.get_octets(), [])


class IPAddressTestGetBinary(unittest.TestCase):
    def test_get_binary_1(self):
        ipaddress = IPAddress("10.10.10.10")
        self.assertEqual(ipaddress.get_binary(), "00001010.00001010.00001010.00001010")

    def test_get_binary_2(self):
        ipaddress = IPAddress("a.10.10.10")
        self.assertEqual(ipaddress.get_binary(), '')

    def test_get_binary_3(self):
        ipaddress = IPAddress("-1.10.10.10")
        self.assertEqual(ipaddress.get_binary(), '')

    def test_get_binary_4(self):
        ipaddress = IPAddress("300.10.10.10")
        self.assertEqual(ipaddress.get_binary(), '')

    def test_get_binary_5(self):
        ipaddress = IPAddress(".10.10.10")
        self.assertEqual(ipaddress.get_binary(), '')


class IPAddressTest(unittest.TestCase):
    def test_IPAddress(self):
        ipaddress = IPAddress("10.10.10.10")
        self.assertEqual(ipaddress.is_valid(), True)
        self.assertEqual(ipaddress.get_octets(), ["10", "10", "10", "10"])
        self.assertEqual(ipaddress.get_binary(), "00001010.00001010.00001010.00001010")

