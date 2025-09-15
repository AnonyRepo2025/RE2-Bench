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
       jsonl_path = json_base + "/JSONProcessor.jsonl"
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
# This is a class to process JSON file, including reading and writing JSON files, as well as processing JSON data by removing a specified key from the JSON object.

import json
import os

class JSONProcessor:
    def read_json(self, file_path):
        """
        Read a JSON file and return the data.
        :param file_path: str, the path of the JSON file.
        :return: dict, the data from the JSON file if read successfully, or return -1 if an error occurs during the reading process.
                    return 0 if the file does not exist.
        >>> json.read_json('test.json')
        {'name': 'test', 'age': 14}
        """

    def write_json(self, data, file_path):
        """
        Write data to a JSON file and save it to the given path.

        :param data: dict, the data to be written to the JSON file.
        :param file_path: str, the path of the JSON file.
        :return: 1 if the writing process is successful, or -1, if an error occurs during the writing process.
        >>> json.write_json({'key1': 'value1', 'key2': 'value2'}, 'test.json')
        1
        >>> json.read_json('test.json')
        {'key1': 'value1', 'key2': 'value2'}
        """

    def process_json(self, file_path, remove_key):
        """
        read a JSON file and process the data by removing a specified key and rewrite the modified data back to the file.

        :param file_path: str, the path of the JSON file.
        :param remove_key: str, the key to be removed.
        :return: 1, if the specified key is successfully removed and the data is written back.
                    0, if the file does not exist or the specified key does not exist in the data.
        >>> json.read_json('test.json')
        {'key1': 'value1', 'key2': 'value2'}
        >>> json.process_json('test.json', 'key1')
        1
        >>> json.read_json('test.json')
        {'key2': 'value2'}
        """
'''

import json
import os


class JSONProcessor:
    @inspect_code
    def read_json(self, file_path):
        if not os.path.exists(file_path):
            return 0
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except:
            return -1

    @inspect_code
    def write_json(self, data, file_path):
        try:
            with open(file_path, 'w') as file:
                json.dump(data, file)
            return 1
        except:
            return -1

    @inspect_code
    def process_json(self, file_path, remove_key):
        data = self.read_json(file_path)
        if data == 0 or data == -1:
            return 0
        if remove_key in data:
            del data[remove_key]
            self.write_json(data, file_path)
            return 1
        else:
            return 0

import os
import stat
import json
import unittest


class JSONProcessorTestReadJson(unittest.TestCase):
    def setUp(self):
        self.processor = JSONProcessor()
        self.test_data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        self.file_path = "test.json"

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    # file exists
    def test_read_json_1(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.test_data, file)
        result = self.processor.read_json(self.file_path)
        self.assertEqual(result, self.test_data)

    # file not exists
    def test_read_json_2(self):
        result = self.processor.read_json(self.file_path)
        self.assertEqual(result, 0)

    # invalid json file
    def test_read_json_3(self):
        with open(self.file_path, 'w') as file:
            file.write("Invalid JSON")
        result = self.processor.read_json(self.file_path)
        self.assertEqual(result, -1)

    def test_read_json_4(self):
        result = self.processor.read_json('wrong')
        self.assertEqual(result, 0)

    def test_read_json_5(self):
        result = self.processor.read_json('abcd')
        self.assertEqual(result, 0)


class JSONProcessorTestWriteJson(unittest.TestCase):
    def setUp(self):
        self.processor = JSONProcessor()
        self.test_data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        self.file_path = "test.json"

        # create a read only file
        self.file_path_only_read = 'test_only_read.json'
        with open(self.file_path_only_read, 'w') as f:
            f.write('{"key1": "value1"}')

        # set file only read mode
        os.chmod(self.file_path_only_read, stat.S_IRUSR + stat.S_IRGRP + stat.S_IROTH)

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        if os.path.exists(self.file_path_only_read):
            # unset file only read mode and remove the file
            os.chmod(self.file_path_only_read,
                     stat.S_IWUSR + stat.S_IRUSR + stat.S_IWGRP + stat.S_IRGRP + stat.S_IWOTH + stat.S_IROTH)
            os.remove(self.file_path_only_read)

    def test_write_json_1(self):
        result = self.processor.write_json(self.test_data, self.file_path)
        self.assertEqual(result, 1)
        with open(self.file_path, 'r') as file:
            written_data = json.load(file)
        self.assertEqual(written_data, self.test_data)

    def test_write_json_2(self):
        # Provide a read-only file path to simulate an exception
        result = self.processor.write_json(self.test_data, self.file_path_only_read)
        self.assertEqual(result, -1)

    def test_write_json_3(self):
        result = self.processor.write_json([], self.file_path_only_read)
        self.assertEqual(result, -1)

    def test_write_json_4(self):
        result = self.processor.write_json(self.test_data, '')
        self.assertEqual(result, -1)

    def test_write_json_5(self):
        result = self.processor.write_json([], '')
        self.assertEqual(result, -1)


class JSONProcessorTestProcessJsonExistingKey(unittest.TestCase):
    def setUp(self):
        self.processor = JSONProcessor()
        self.test_data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        self.file_path = "test.json"

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    # key exists
    def test_process_json_1(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.test_data, file)
        remove_key = "key2"
        self.processor.process_json(self.file_path, remove_key)
        with open(self.file_path, 'r') as file:
            processed_data = json.load(file)
        expected_data = {
            "key1": "value1",
            "key3": "value3"
        }
        self.assertEqual(processed_data, expected_data)

    # key not exists
    def test_process_json_2(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.test_data, file)
        remove_key = "nonexistent_key"
        self.processor.process_json(self.file_path, remove_key)
        with open(self.file_path, 'r') as file:
            processed_data = json.load(file)
        self.assertEqual(processed_data, self.test_data)

    # file is empty
    def test_process_json_3(self):
        # Create an empty JSON file
        with open(self.file_path, 'w') as file:
            pass
        remove_key = "key1"
        self.assertEqual(self.processor.process_json(self.file_path, remove_key), 0)

    def test_process_json_4(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.test_data, file)
        remove_key = "aaa"
        self.processor.process_json(self.file_path, remove_key)
        with open(self.file_path, 'r') as file:
            processed_data = json.load(file)
        self.assertEqual(processed_data, self.test_data)

    def test_process_json_5(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.test_data, file)
        remove_key = "bbb"
        self.processor.process_json(self.file_path, remove_key)
        with open(self.file_path, 'r') as file:
            processed_data = json.load(file)
        self.assertEqual(processed_data, self.test_data)


class JSONProcessorTestMain(unittest.TestCase):
    def setUp(self):
        self.processor = JSONProcessor()
        self.test_data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        self.file_path = "test.json"

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_main(self):
        # write first
        result = self.processor.write_json(self.test_data, self.file_path)
        self.assertEqual(result, 1)
        with open(self.file_path, 'r') as file:
            written_data = json.load(file)
        self.assertEqual(written_data, self.test_data)

        # read
        result = self.processor.read_json(self.file_path)
        self.assertEqual(result, self.test_data)

        # process
        remove_key = "key2"
        self.processor.process_json(self.file_path, remove_key)
        with open(self.file_path, 'r') as file:
            processed_data = json.load(file)
        expected_data = {
            "key1": "value1",
            "key3": "value3"
        }
        self.assertEqual(processed_data, expected_data)

