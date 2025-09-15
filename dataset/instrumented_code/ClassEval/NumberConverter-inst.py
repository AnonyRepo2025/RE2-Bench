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
       jsonl_path = json_base + "/NumberConverter.jsonl"
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
# The class allows to convert  decimal to binary, octal and hexadecimal repectively and contrarily

class NumberConverter:
    @staticmethod
    def decimal_to_binary(decimal_num):
        """
        Convert a number from decimal format to binary format.
        :param decimal_num: int, decimal number
        :return: str, the binary representation of an integer.
        >>> NumberConverter.decimal_to_binary(42423)
        '1010010110110111'
        """

    @staticmethod
    def binary_to_decimal(binary_num):
        """
        Convert a number from binary format to decimal format.
        :param binary_num: str, binary number
        :return: int, the decimal representation of binary number str.
        >>> NumberConverter.binary_to_decimal('1010010110110111')
        42423
        """


    @staticmethod
    def decimal_to_octal(decimal_num):
        """
        Convert a number from decimal format to octal format.
        :param decimal_num: int, decimal number
        :return: str, the octal representation of an integer.
        >>> NumberConverter.decimal_to_octal(42423)
        '122667'
        """

    @staticmethod
    def octal_to_decimal(octal_num):
        """
        Convert a number from octal format to decimal format.
        :param octal_num: str, octal num
        :return: int, the decimal representation of octal number str.
        >>> NumberConverter.octal_to_decimal('122667')
        42423
        """

    @staticmethod
    def decimal_to_hex(decimal_num):
        """
        Convert a number from decimal format to hex format.
        :param decimal_num: int, decimal number
        :return hex_num: str, the hex representation of an integer.
        >>> NumberConverter.decimal_to_hex(42423)
        'a5b7'
        """

    @staticmethod
    def hex_to_decimal(hex_num):
        """
        Convert a number from hex format to decimal format.
        :param hex_num: str, hex num
        :return: int, the decimal representation of hex number str.
        >>> NumberConverter.hex_to_decimal('a5b7')
        42423
        """

'''


class NumberConverter:
    @staticmethod
    @inspect_code
    def decimal_to_binary(decimal_num):
        binary_num = bin(decimal_num)[2:]
        return binary_num

    @staticmethod
    @inspect_code
    def binary_to_decimal(binary_num):
        decimal_num = int(binary_num, 2)
        return decimal_num

    @staticmethod
    @inspect_code
    def decimal_to_octal(decimal_num):
        octal_num = oct(decimal_num)[2:]
        return octal_num

    @staticmethod
    @inspect_code
    def octal_to_decimal(octal_num):
        decimal_num = int(octal_num, 8)
        return decimal_num

    @staticmethod
    @inspect_code
    def decimal_to_hex(decimal_num):
        hex_num = hex(decimal_num)[2:]
        return hex_num

    @staticmethod
    @inspect_code
    def hex_to_decimal(hex_num):
        decimal_num = int(hex_num, 16)
        return decimal_num

import unittest


class NumberConverterTestDecimalToBinary(unittest.TestCase):
    def test_decimal_to_binary(self):
        self.assertEqual('1010010110110111', NumberConverter.decimal_to_binary(42423))

    def test_decimal_to_binary_2(self):
        self.assertEqual('101001100010111', NumberConverter.decimal_to_binary(21271))

    def test_decimal_to_binary_3(self):
        self.assertEqual('1010010111010111', NumberConverter.decimal_to_binary(42455))

    def test_decimal_to_binary_4(self):
        self.assertEqual('10100101110101011', NumberConverter.decimal_to_binary(84907))

    def test_decimal_to_binary_5(self):
        self.assertEqual('101001011101010111', NumberConverter.decimal_to_binary(169815))

class NumberConverterTestBinaryToDecimal(unittest.TestCase):
    def test_binary_to_decimal(self):
        self.assertEqual(42423, NumberConverter.binary_to_decimal('1010010110110111'))

    def test_binary_to_decimal_2(self):
        self.assertEqual(10615, NumberConverter.binary_to_decimal('10100101110111'))

    def test_binary_to_decimal_3(self):
        self.assertEqual(42455, NumberConverter.binary_to_decimal('1010010111010111'))

    def test_binary_to_decimal_4(self):
        self.assertEqual(169819, NumberConverter.binary_to_decimal('101001011101011011'))

    def test_binary_to_decimal_5(self):
        self.assertEqual(339639, NumberConverter.binary_to_decimal('1010010111010110111'))

class NumberConvertTestDecimalToOctal(unittest.TestCase):
    def test_decimal_to_octal(self):
        self.assertEqual('122667', NumberConverter.decimal_to_octal(42423))

    def test_decimal_to_octal_2(self):
        self.assertEqual('51427', NumberConverter.decimal_to_octal(21271))

    def test_decimal_to_octal_3(self):
        self.assertEqual('245653', NumberConverter.decimal_to_octal(84907))

    def test_decimal_to_octal_4(self):
        self.assertEqual('513527', NumberConverter.decimal_to_octal(169815))

    def test_decimal_to_octal_5(self):
        self.assertEqual('1227256', NumberConverter.decimal_to_octal(339630))

class NumberConvertTestOctalToDecimal(unittest.TestCase):
    def test_octal_to_decimal(self):
        self.assertEqual(42423, NumberConverter.octal_to_decimal('122667'))

    def test_octal_to_decimal_2(self):
        self.assertEqual(21271, NumberConverter.octal_to_decimal('51427'))

    def test_octal_to_decimal_3(self):
        self.assertEqual(84907, NumberConverter.octal_to_decimal('245653'))

    def test_octal_to_decimal_4(self):
        self.assertEqual(169815, NumberConverter.octal_to_decimal('513527'))

    def test_octal_to_decimal_5(self):
        self.assertEqual(339630, NumberConverter.octal_to_decimal('1227256'))

class NumberConvertTestDecimalToHex(unittest.TestCase):
    def test_decimal_to_hex(self):
        self.assertEqual('a5b7', NumberConverter.decimal_to_hex(42423))

    def test_decimal_to_hex_2(self):
        self.assertEqual('5317', NumberConverter.decimal_to_hex(21271))

    def test_decimal_to_hex_3(self):
        self.assertEqual('14bab', NumberConverter.decimal_to_hex(84907))

    def test_decimal_to_hex_4(self):
        self.assertEqual('29757', NumberConverter.decimal_to_hex(169815))

    def test_decimal_to_hex_5(self):
        self.assertEqual('52eb7', NumberConverter.decimal_to_hex(339639))

class NumberConvertTestHexToDecimal(unittest.TestCase):
    def test_hex_to_decimal(self):
        self.assertEqual(42423, NumberConverter.hex_to_decimal('a5b7'))

    def test_hex_to_decimal_2(self):
        self.assertEqual(21207, NumberConverter.hex_to_decimal('52d7'))

    def test_hex_to_decimal_3(self):
        self.assertEqual(84627, NumberConverter.hex_to_decimal('14a93'))

    def test_hex_to_decimal_4(self):
        self.assertEqual(170615, NumberConverter.hex_to_decimal('29a77'))

    def test_hex_to_decimal_5(self):
        self.assertEqual(342647, NumberConverter.hex_to_decimal('53a77'))

class NumberConvertTestMain(unittest.TestCase):
    def test_main(self):
        self.assertEqual('1010010110110111', NumberConverter.decimal_to_binary(42423))
        self.assertEqual(42423, NumberConverter.binary_to_decimal('1010010110110111'))
        self.assertEqual('122667', NumberConverter.decimal_to_octal(42423))
        self.assertEqual('122667', NumberConverter.decimal_to_octal(42423))
        self.assertEqual('a5b7', NumberConverter.decimal_to_hex(42423))
        self.assertEqual(42423, NumberConverter.hex_to_decimal('a5b7'))