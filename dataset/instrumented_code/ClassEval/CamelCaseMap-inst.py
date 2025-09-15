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
       jsonl_path = json_base + "/CamelCaseMap.jsonl"
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
# This is a custom class that allows keys to be in camel case style by converting them from underscore style, which provides dictionary-like functionality.

class CamelCaseMap:
    def __init__(self):
        """
        Initialize data to an empty dictionary
        """
        self._data = {}

    def __getitem__(self, key):
        """
        Return the value corresponding to the key
        :param key:str
        :return:str,the value corresponding to the key
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map['first_name'] = 'John'
        >>> camelize_map.__getitem__('first_name')
        'John'
        """


    def __setitem__(self, key, value):
        """
        Set the value corresponding to the key to the specified value
        :param key:str
        :param value:str, the specified value
        :return:None
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map['first_name'] = 'John'
        >>> camelize_map.__setitem__('first_name', 'new name')
        camelize_map['first_name'] = 'new name'
        """


    def __delitem__(self, key):
        """
        Delete the value corresponding to the key
        :param key:str
        :return:None
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map['first_name'] = 'John'
        >>> camelize_map.__delitem__('first_name')
        >>> flag = 'first_name' in camelize_map
        flag = False
        """


    def __iter__(self):
        """
        Returning Iterateable Objects with Own Data
        :return:Iterator
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map['first_name'] = 'John'
        >>> camelize_map['last_name'] = 'Doe'
        >>> camelize_map['age'] = 30
        >>> camelize_map.__iter__()
        <dict_keyiterator object at 0x0000026739977C20>
        """


    def __len__(self):
        """
        Returns the length of the own data
        :return:int, length of data
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map['first_name'] = 'John'
        >>> camelize_map['last_name'] = 'Doe'
        >>> camelize_map['age'] = 30
        >>> camelize_map.__len__()
        3
        """

    def _convert_key(self, key):
        """
        convert key string into camel case
        :param key:str
        :return:str, converted key string
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map._convert_key('first_name')
        'firstName'
        """

    @staticmethod
    def _to_camel_case(key):
        """
        convert key string into camel case
        :param key:str
        :return:str, converted key string
        >>> camelize_map = CamelCaseMap()
        >>> camelize_map._to_camel_case('first_name')
        'firstName'
        """

'''


class CamelCaseMap:
    def __init__(self):
        self._data = {}

    @inspect_code
    def __getitem__(self, key):
        return self._data[self._convert_key(key)]

    @inspect_code
    def __setitem__(self, key, value):
        self._data[self._convert_key(key)] = value

    @inspect_code
    def __delitem__(self, key):
        del self._data[self._convert_key(key)]

    @inspect_code
    def __iter__(self):
        return iter(self._data)

    @inspect_code
    def __len__(self):
        return len(self._data)

    @inspect_code
    def _convert_key(self, key):
        if isinstance(key, str):
            return self._to_camel_case(key)
        return key

    @staticmethod
    @inspect_code
    def _to_camel_case(key):
        parts = key.split('_')
        return parts[0] + ''.join(part.title() for part in parts[1:])



import unittest


class CamelCaseMapTestGetitem(unittest.TestCase):
    def test_getitem_1(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        self.assertEqual(camelize_map.__getitem__('first_name'), 'John')

    def test_getitem_2(self):
        camelize_map = CamelCaseMap()
        camelize_map['last_name'] = 'Doe'
        self.assertEqual(camelize_map.__getitem__('last_name'), 'Doe')

    def test_getitem_3(self):
        camelize_map = CamelCaseMap()
        camelize_map['age'] = 30
        self.assertEqual(camelize_map.__getitem__('age'), 30)

    def test_getitem_4(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        self.assertEqual(camelize_map.__getitem__('first_Name'), 'John')

    def test_getitem_5(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        self.assertEqual(camelize_map.__getitem__('firstName'), 'John')


class CamelCaseMapTestSetitem(unittest.TestCase):
    def test_setitem_1(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map.__setitem__('first_name', 'newname')
        self.assertEqual(camelize_map['first_name'], 'newname')

    def test_setitem_2(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map.__setitem__('first_name', 'John')
        self.assertEqual(camelize_map['first_name'], 'John')

    def test_setitem_3(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map.__setitem__('first_Name', 'newname')
        self.assertEqual(camelize_map['first_name'], 'newname')

    def test_setitem_4(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map.__setitem__('firstName', 'newname')
        self.assertEqual(camelize_map['first_name'], 'newname')

    def test_setitem_5(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map.__setitem__('first_name', '')
        self.assertEqual(camelize_map['first_name'], '')


class CamelCaseMapTestDelitem(unittest.TestCase):
    def test_delitem_1(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map['last_name'] = 'Doe'
        camelize_map.__delitem__('first_name')
        self.assertEqual(camelize_map['last_name'], 'Doe')

    def test_delitem_2(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map.__delitem__('first_name')
        self.assertEqual('first_name' in camelize_map, False)

    def test_delitem_3(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map.__delitem__('first_Name')
        self.assertEqual('first_name' in camelize_map, False)

    def test_delitem_4(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map.__delitem__('firstName')
        self.assertEqual('first_name' in camelize_map, False)

    def test_delitem_5(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = ''
        camelize_map.__delitem__('first_name')
        self.assertEqual('first_name' in camelize_map, False)


class CamelCaseMapTestIter(unittest.TestCase):
    def test_iter_1(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map['last_name'] = 'Doe'
        camelize_map['age'] = 30
        lst = ['firstName', 'lastName', 'age']
        iter = camelize_map.__iter__()
        i = 0
        for key in iter:
            self.assertEqual(key, lst[i])
            i += 1

    def test_iter_2(self):
        camelize_map = CamelCaseMap()
        camelize_map['firstname'] = 'John'
        camelize_map['lastname'] = 'Doe'
        camelize_map['age'] = 30
        lst = ['firstname', 'lastname', 'age']
        iter = camelize_map.__iter__()
        i = 0
        for key in iter:
            self.assertEqual(key, lst[i])
            i += 1

    def test_iter_3(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_Name'] = 'John'
        camelize_map['last_Name'] = 'Doe'
        camelize_map['age'] = 30
        lst = ['firstName', 'lastName', 'age']
        iter = camelize_map.__iter__()
        i = 0
        for key in iter:
            self.assertEqual(key, lst[i])
            i += 1

    def test_iter_4(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_Name'] = 'John'
        camelize_map['last_Name'] = 'Doe'
        lst = ['firstName', 'lastName']
        iter = camelize_map.__iter__()
        i = 0
        for key in iter:
            self.assertEqual(key, lst[i])
            i += 1

    def test_iter_5(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_Name'] = 'John'
        lst = ['firstName']
        iter = camelize_map.__iter__()
        i = 0
        for key in iter:
            self.assertEqual(key, lst[i])
            i += 1


class CamelCaseMapTestLen(unittest.TestCase):
    def test_len_1(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        self.assertEqual(camelize_map.__len__(), 1)

    def test_len_2(self):
        camelize_map = CamelCaseMap()
        camelize_map['last_name'] = 'Doe'
        self.assertEqual(camelize_map.__len__(), 1)

    def test_len_3(self):
        camelize_map = CamelCaseMap()
        camelize_map['age'] = 30
        self.assertEqual(camelize_map.__len__(), 1)

    def test_len_4(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map['last_Name'] = 'Doe'
        camelize_map['age'] = 30
        self.assertEqual(camelize_map.__len__(), 3)

    def test_len_5(self):
        camelize_map = CamelCaseMap()
        self.assertEqual(camelize_map.__len__(), 0)


class CamelCaseMapTestConvertKey(unittest.TestCase):
    def test_convert_key_1(self):
        camelize_map = CamelCaseMap()
        self.assertEqual(camelize_map._convert_key('aaa_bbb'), 'aaaBbb')

    def test_convert_key_2(self):
        camelize_map = CamelCaseMap()
        self.assertEqual(camelize_map._convert_key('first_name'), 'firstName')

    def test_convert_key_3(self):
        camelize_map = CamelCaseMap()
        self.assertEqual(camelize_map._convert_key('last_name'), 'lastName')

    def test_convert_key_4(self):
        camelize_map = CamelCaseMap()
        self.assertEqual(camelize_map._convert_key('ccc_ddd'), 'cccDdd')

    def test_convert_key_5(self):
        camelize_map = CamelCaseMap()
        self.assertEqual(camelize_map._convert_key('eee_fff'), 'eeeFff')

    def test_convert_key_6(self):
        camelize_map = CamelCaseMap()
        self.assertEqual(camelize_map._convert_key(1234), 1234)


class CamelCaseMapTestToCamelCase(unittest.TestCase):
    def test_to_camel_case_1(self):
        self.assertEqual(CamelCaseMap._to_camel_case('aaa_bbb'), 'aaaBbb')

    def test_to_camel_case_2(self):
        self.assertEqual(CamelCaseMap._to_camel_case('first_name'), 'firstName')

    def test_to_camel_case_3(self):
        self.assertEqual(CamelCaseMap._to_camel_case('last_name'), 'lastName')

    def test_to_camel_case_4(self):
        self.assertEqual(CamelCaseMap._to_camel_case('ccc_ddd'), 'cccDdd')

    def test_to_camel_case_5(self):
        self.assertEqual(CamelCaseMap._to_camel_case('eee_fff'), 'eeeFff')


class CamelCaseMapTest(unittest.TestCase):
    def test_CamelCaseMap(self):
        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        self.assertEqual(camelize_map.__getitem__('first_name'), 'John')

        camelize_map = CamelCaseMap()
        camelize_map['first_name'] = 'John'
        camelize_map.__setitem__('first_name', 'newname')
        self.assertEqual(camelize_map['first_name'], 'newname')
