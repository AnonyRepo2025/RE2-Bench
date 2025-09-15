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
       jsonl_path = json_base + "/CSVProcessor.jsonl"
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
# This is a class for processing CSV files, including readring and writing CSV data, as well as processing specific operations and saving as a new CSV file.

import csv

class CSVProcessor:

    def __init__(self):
        pass

    def read_csv(self, file_name):
        """
        Read the csv file by file_name, get the title and data from it
        :param file_name: str, name of the csv file
        :return title, data: (list, list), first row is title, the rest is data
        >>> csvProcessor = CSVProcessor()
        >>> csvProcessor.read_csv('read_test.csv')
        (['a', 'b', 'c', 'd'], [['hElLo', 'YoU', 'ME', 'LoW']])
        """

    def write_csv(self, data, file_name):
        """
        Write data into a csv file.
        :param file_name: str, name of the csv file
        :return:int, if success return 1, or 0 otherwise
        >>> csvProcessor = CSVProcessor()
        >>> csvProcessor.write_csv([['a', 'b', 'c', 'd'], ['1', '2', '3', '4']], 'write_test.csv')
        1
        """

    def process_csv_data(self, N, save_file_name):
        """
        Read a csv file into variable title and data.
        Only remain the N th (from 0) column data and Capitalize them, store the title and new data into a new csv file.
        Add '_process' suffix after old file name, as a new file name.
        :param N: int, the N th column(from 0)
        :param save_file_name, the name of file that needs to be processed.
        :return:int, if success return 1, or 0 otherwise
        >>> csvProcessor = CSVProcessor()
        >>> csvProcessor.read_csv('read_test.csv')
        (['a', 'b', 'c', 'd'], [['hElLo', 'YoU', 'ME', 'LoW']])
        >>> csvProcessor.process_csv_data(0, 'read_test.csv')
        1
        >>> csvProcessor.read_csv('read_test_process.csv')
        (['a', 'b', 'c', 'd'], [['HELLO']])
        """
'''

import csv


class CSVProcessor:

    def __init__(self):
        pass

    @inspect_code
    def read_csv(self, file_name):
        data = []
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            title = next(reader)
            for row in reader:
                data.append(row)
        return title, data

    @inspect_code
    def write_csv(self, data, file_name):
        try:
            with open(file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)
            return 1
        except:
            return 0

    @inspect_code
    def process_csv_data(self, N, save_file_name):
        title, data = self.read_csv(save_file_name)
        column_data = [row[N] for row in data]
        column_data = [row.upper() for row in column_data]
        new_data = [title, column_data]
        return self.write_csv(new_data, save_file_name.split('.')[0] + '_process.csv')

import unittest
import os


class CSVProcessorTestReadCSV(unittest.TestCase):
    def test_read_csv_1(self):
        self.file = 'read_test.csv'

        with open(self.file, 'w') as f:
            f.write('a,b,c,d\nhElLo,YoU,ME,LoW')

        expected_title = ['a', 'b', 'c', 'd']
        expected_data = [['hElLo', 'YoU', 'ME', 'LoW']]
        csvProcessor = CSVProcessor()
        title, data = csvProcessor.read_csv(self.file)
        self.assertEqual(expected_data, data)
        self.assertEqual(expected_title, title)

    def test_read_csv_2(self):
        self.file = 'read_test.csv'
        with open(self.file, 'w') as f:
            f.write('1234\nhElLo,YoU,ME,LoW')

        expected_title = ['1234']
        expected_data = [['hElLo', 'YoU', 'ME', 'LoW']]
        csvProcessor = CSVProcessor()
        title, data = csvProcessor.read_csv(self.file)
        self.assertEqual(expected_data, data)
        self.assertEqual(expected_title, title)

    def test_read_csv_3(self):
        self.file = 'read_test.csv'

        with open(self.file, 'w') as f:
            f.write('title\nhElLo,YoU,ME,LoW')

        expected_title = ['title']
        expected_data = [['hElLo', 'YoU', 'ME', 'LoW']]
        csvProcessor = CSVProcessor()
        title, data = csvProcessor.read_csv(self.file)
        self.assertEqual(expected_data, data)
        self.assertEqual(expected_title, title)

    def test_read_csv_4(self):
        self.file = 'read_test.csv'

        with open(self.file, 'w') as f:
            f.write('title4\nhElLo,YoU,ME,LoW')

        expected_title = ['title4']
        expected_data = [['hElLo', 'YoU', 'ME', 'LoW']]
        csvProcessor = CSVProcessor()
        title, data = csvProcessor.read_csv(self.file)
        self.assertEqual(expected_data, data)
        self.assertEqual(expected_title, title)

    def test_read_csv_5(self):
        self.file = 'read_test.csv'

        with open(self.file, 'w') as f:
            f.write('title5\nhElLo,YoU,ME,LoW')

        expected_title = ['title5']
        expected_data = [['hElLo', 'YoU', 'ME', 'LoW']]
        csvProcessor = CSVProcessor()
        title, data = csvProcessor.read_csv(self.file)
        self.assertEqual(expected_data, data)
        self.assertEqual(expected_title, title)


class CSVProcessorTestWriteCSV(unittest.TestCase):
    def test_write_csv_1(self):
        self.file = 'read_test.csv'

        file_path = self.file
        csvProcessor = CSVProcessor()
        data = [['a', 'b', 'c', 'd'], ['1', '2', '3', '4']]
        # assert return value
        self.assertEqual(1, csvProcessor.write_csv(data, file_path))

        # read to test if write correctly
        read_title, read_data = csvProcessor.read_csv(file_path)
        self.assertEqual(read_title, data[0])
        self.assertEqual(read_data[0], data[1])
        os.remove(file_path)

    def test_write_csv_2(self):
        self.file = 'read_test.csv'

        file_path = self.file
        csvProcessor = CSVProcessor()
        data = [['aa', 'bb', 'cc', 'dd'], ['1', '2', '3', '4']]
        # assert return value
        self.assertEqual(1, csvProcessor.write_csv(data, file_path))

        # read to test if write correctly
        read_title, read_data = csvProcessor.read_csv(file_path)
        self.assertEqual(read_title, data[0])
        self.assertEqual(read_data[0], data[1])
        os.remove(file_path)

    def test_write_csv_3(self):
        self.file = 'read_test.csv'

        file_path = self.file
        csvProcessor = CSVProcessor()
        data = [['a', 'b', 'c', 'd'], ['11', '22', '33', '44']]
        # assert return value
        self.assertEqual(1, csvProcessor.write_csv(data, file_path))

        # read to test if write correctly
        read_title, read_data = csvProcessor.read_csv(file_path)
        self.assertEqual(read_title, data[0])
        self.assertEqual(read_data[0], data[1])
        os.remove(file_path)

    def test_write_csv_4(self):
        self.file = 'read_test.csv'

        file_path = self.file
        csvProcessor = CSVProcessor()
        data = [['e', 'f', 'g', 'h'], ['1', '2', '3', '4']]
        # assert return value
        self.assertEqual(1, csvProcessor.write_csv(data, file_path))

        # read to test if write correctly
        read_title, read_data = csvProcessor.read_csv(file_path)
        self.assertEqual(read_title, data[0])
        self.assertEqual(read_data[0], data[1])
        os.remove(file_path)

    def test_write_csv_5(self):
        self.file = 'read_test.csv'

        file_path = self.file
        csvProcessor = CSVProcessor()
        data = [['a', 'b', 'c', 'd'], ['5', '6', '7', '8']]
        # assert return value
        self.assertEqual(1, csvProcessor.write_csv(data, file_path))

        # read to test if write correctly
        read_title, read_data = csvProcessor.read_csv(file_path)
        self.assertEqual(read_title, data[0])
        self.assertEqual(read_data[0], data[1])
        os.remove(file_path)

    def test_write_csv_6(self):
        self.file = ''
        file_path = self.file
        csvProcessor = CSVProcessor()
        # assert return value
        self.assertEqual(0, csvProcessor.write_csv([], file_path))


class CSVProcessorTestProcessCSVData(unittest.TestCase):
    def setUp(self) -> None:
        self.file = 'read_test.csv'
        self.file_process = 'read_test_process.csv'
        with open(self.file, 'w') as f:
            f.write('a,b,c,d\nhElLo,YoU,ME,LoW,aBc')

    def test_process_csv_data_1(self):
        title = ['a', 'b', 'c', 'd']
        data = ['HELLO']
        csvProcessor = CSVProcessor()
        self.assertEqual(1, csvProcessor.process_csv_data(0, self.file))

        read_title, read_data = csvProcessor.read_csv(self.file_process)
        self.assertEqual(read_title, title)
        self.assertEqual(read_data[0], data)

    def test_process_csv_data_2(self):
        title = ['a', 'b', 'c', 'd']
        data = ['YOU']
        csvProcessor = CSVProcessor()
        self.assertEqual(1, csvProcessor.process_csv_data(1, self.file))

        read_title, read_data = csvProcessor.read_csv(self.file_process)
        self.assertEqual(read_title, title)
        self.assertEqual(read_data[0], data)

    def test_process_csv_data_3(self):
        title = ['a', 'b', 'c', 'd']
        data = ['ME']
        csvProcessor = CSVProcessor()
        self.assertEqual(1, csvProcessor.process_csv_data(2, self.file))

        read_title, read_data = csvProcessor.read_csv(self.file_process)
        self.assertEqual(read_title, title)
        self.assertEqual(read_data[0], data)

    def test_process_csv_data_4(self):
        title = ['a', 'b', 'c', 'd']
        data = ['LOW']
        csvProcessor = CSVProcessor()
        self.assertEqual(1, csvProcessor.process_csv_data(3, self.file))

        read_title, read_data = csvProcessor.read_csv(self.file_process)
        self.assertEqual(read_title, title)
        self.assertEqual(read_data[0], data)

    def test_process_csv_data_5(self):
        title = ['a', 'b', 'c', 'd']
        data = ['ABC']
        csvProcessor = CSVProcessor()
        self.assertEqual(1, csvProcessor.process_csv_data(4, self.file))

        read_title, read_data = csvProcessor.read_csv(self.file_process)
        self.assertEqual(read_title, title)
        self.assertEqual(read_data[0], data)


class CSVProcessorTestMain(unittest.TestCase):
    def setUp(self) -> None:
        self.file = 'test.csv'
        self.file_process = 'test_process.csv'
        with open(self.file, 'w') as f:
            f.write('a,b,c,d\nhElLo,YoU,ME,LoW')

    def test_main(self):
        csvProcessor = CSVProcessor()
        data = [['a', 'b', 'c', 'd'], ['hElLo', 'YoU', 'ME', 'LoW']]
        # assert return value
        self.assertEqual(1, csvProcessor.write_csv(data, self.file))
        expected_title = ['a', 'b', 'c', 'd']
        expected_data = [['hElLo', 'YoU', 'ME', 'LoW']]
        title, data = csvProcessor.read_csv(self.file)
        self.assertEqual(expected_data, data)
        self.assertEqual(expected_title, title)

        title = ['a', 'b', 'c', 'd']
        data = ['HELLO']
        self.assertEqual(1, csvProcessor.process_csv_data(0, self.file))

        read_title, read_data = csvProcessor.read_csv(self.file_process)
        self.assertEqual(read_title, title)
        self.assertEqual(read_data[0], data)
