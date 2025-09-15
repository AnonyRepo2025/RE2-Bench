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
       jsonl_path = json_base + "/NLPDataProcessor2.jsonl"
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
# The class processes NLP data by extracting words from a list of strings, calculating the frequency of each word, and returning the top 5 most frequent words.

import re
from collections import Counter

class NLPDataProcessor2:

    def process_data(self, string_list):
        """
        keep only English letters and spaces in the string, then convert the string to lower case, and then split the string into a list of words.
        :param string_list: a list of strings
        :return: words_list: a list of words lists
        >>> NLPDataProcessor.process_data(['This is a test.'])
        [['this', 'is', 'a', 'test']]
        """

    def calculate_word_frequency(self, words_list):
        """
        Calculate the word frequency of each word in the list of words list, and sort the word frequency dictionary by value in descending order.
        :param words_list: a list of words lists
        :return: top 5 word frequency dictionary, a dictionary of word frequency, key is word, value is frequency
        >>> NLPDataProcessor.calculate_word_frequency([['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']])
        {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}
        """

    def process(self, string_list):
        """
        keep only English letters and spaces in the string, then convert the string to lower case, and then split the string into a list of words. Calculate the word frequency of each word in the list of words list, and sort the word frequency dictionary by value in descending order.
        :param string_list: a list of strings
        :return: top 5 word frequency dictionary, a dictionary of word frequency, key is word, value is frequency
        >>> NLPDataProcessor.process(['This is a test.', 'This is another test.'])
        {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}
        """

'''

from collections import Counter
import re

class NLPDataProcessor2:

    @inspect_code
    def process_data(self, string_list):
        words_list = []
        for string in string_list:
            # Remove non-English letters and convert to lowercase
            processed_string = re.sub(r'[^a-zA-Z\s]', '', string.lower())
            # Split the string into words
            words = processed_string.split()
            words_list.append(words)
        return words_list

    @inspect_code
    def calculate_word_frequency(self, words_list):
        word_frequency = Counter()
        for words in words_list:
            word_frequency.update(words)
        sorted_word_frequency = dict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))
        top_5_word_frequency = dict(list(sorted_word_frequency.items())[:5])
        return top_5_word_frequency

    @inspect_code
    def process(self, string_list):
        words_list = self.process_data(string_list)
        word_frequency_dict = self.calculate_word_frequency(words_list)
        return word_frequency_dict


import unittest

class NLPDataProcessorTestProcessData(unittest.TestCase):

    def setUp(self):
        self.processor = NLPDataProcessor2()

    def test_process_data(self):
        string_list = ["Hello World!", "This is a test."]
        expected_output = [['hello', 'world'], ['this', 'is', 'a', 'test']]
        self.assertEqual(self.processor.process_data(string_list), expected_output)

    def test_process_data2(self):
        string_list = ["12345", "Special@Characters"]
        expected_output = [[], ['specialcharacters']]
        self.assertEqual(self.processor.process_data(string_list), expected_output)

    def test_process_data3(self):
        string_list = []
        expected_output = []
        self.assertEqual(self.processor.process_data(string_list), expected_output)

    def test_process_data4(self):
        string_list = ["Hello World!", "This is a test.", "12345", "Special@Characters"]
        expected_output = [['hello', 'world'], ['this', 'is', 'a', 'test'], [], ['specialcharacters']]
        self.assertEqual(self.processor.process_data(string_list), expected_output)

    def test_process_data5(self):
        string_list = ["Hello World!", "This is a test.", "12345", "Special@Characters", "Hello World!", "This is a test.", "12345", "Special@Characters"]
        expected_output = [['hello', 'world'], ['this', 'is', 'a', 'test'], [], ['specialcharacters'], ['hello', 'world'], ['this', 'is', 'a', 'test'], [], ['specialcharacters']]
        self.assertEqual(self.processor.process_data(string_list), expected_output)

class NLPDataProcessorTestCalculate(unittest.TestCase):

    def setUp(self):
        self.processor = NLPDataProcessor2()

    def test_calculate_word_frequency(self):
        words_list = [['hello', 'world'], ['this', 'is', 'a', 'test'], ['hello', 'world', 'this', 'is', 'another', 'test'],
                      ['hello', 'hello', 'world']]
        expected_output = {'hello': 4, 'world': 3, 'this': 2, 'is': 2, 'test': 2}
        self.assertEqual(self.processor.calculate_word_frequency(words_list), expected_output)

    def test_calculate_word_frequency2(self):
        words_list = [['hello', 'world'], ['this', 'is', 'a', 'test'], ['hello', 'world', 'this', 'is', 'another', 'test'],
                      ['hello', 'hello', 'world'], ['world', 'world', 'world']]
        expected_output = {'world': 6, 'hello': 4, 'this': 2, 'is': 2, 'test': 2}
        self.assertEqual(self.processor.calculate_word_frequency(words_list), expected_output)

    def test_calculate_word_frequency3(self):
        words_list = [['hello', 'world'], ['hello', 'hello', 'world'], ['world', 'world']]
        expected_output = {'world': 4, 'hello': 3}
        self.assertEqual(self.processor.calculate_word_frequency(words_list), expected_output)

    def test_calculate_word_frequency4(self):
        words_list = [['hello', 'world'], ['this', 'is', 'a', '%%%'], ['hello', 'world', 'this', 'is', 'another', '%%%'],
                      ['hello', 'hello', 'world'], ['%%%', 'world', 'a', '%%%'], ['%%%', 'hello', '%%%']]
        expected_output = {'%%%': 6, 'hello': 5, 'world': 4, 'is': 2, 'this': 2}
        self.assertEqual(self.processor.calculate_word_frequency(words_list), expected_output)

    def test_calculate_word_frequency5(self):
        words_list = [['hello', 'world'], ['this', 'is', 'a', '%%%'], ['hello', 'world', 'this', 'is', 'another', '%%%'],
                      ['hello', 'hello', 'world'], ['%%%', 'world', 'a', '%%%'], ['%%%', 'hello', '%%%'], ['hello', 'world'], ['this', 'is', 'a', '%%%'], ['hello', 'world', 'this', 'is', 'another', '%%%'],
                      ['hello', 'hello', 'world'], ['%%%', 'world', 'a', '%%%'], ['%%%', 'hello', '%%%']]
        expected_output = {'%%%': 12, 'hello': 10, 'world': 8, 'is': 4, 'this': 4}
        self.assertEqual(self.processor.calculate_word_frequency(words_list), expected_output)

class NLPDataProcessorTestProcess(unittest.TestCase):

    def setUp(self):
        self.processor = NLPDataProcessor2()

    def test_process(self):
        string_list = ["Hello World!", "This is a test.", "Hello World, this is a test."]
        expected_output = {'hello': 2, 'world': 2, 'this': 2, 'is': 2, 'a': 2}
        self.assertEqual(self.processor.process(string_list), expected_output)

    def test_process2(self):
        string_list = []
        expected_output = []
        self.assertEqual(self.processor.process_data(string_list), expected_output)

    def test_calculate3(self):
        words_list = []
        expected_output = {}
        self.assertEqual(self.processor.calculate_word_frequency(words_list), expected_output)

    def test_process4(self):
        string_list = ["@#$%^&*", "Special_Characters", "12345"]
        expected_output = [[], ['specialcharacters'], []]
        self.assertEqual(self.processor.process_data(string_list), expected_output)

    def test_process5(self):
        string_list = ["Hello World! %%%", "This is a %%% test. %%% ", "Hello World, this is a test. %%%"]
        expected_output = {'hello': 2, 'world': 2, 'this': 2, 'is': 2, 'a': 2}
        self.assertEqual(self.processor.process(string_list), expected_output)

    def test_process6(self):
        string_list = ["12345", "67890", "98765"]
        expected_output = [[], [], []]
        self.assertEqual(self.processor.process_data(string_list), expected_output)



