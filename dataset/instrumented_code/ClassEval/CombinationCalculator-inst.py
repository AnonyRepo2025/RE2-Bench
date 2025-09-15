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
       jsonl_path = json_base + "/CombinationCalculator.jsonl"
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
# This is a class that provides methods to calculate the number of combinations for a specific count, calculate all possible combinations, and generate combinations with a specified number of elements.

import math
from typing import List

class CombinationCalculator:
    def __init__(self, datas: List[str]):
        """
        Initialize the calculator with a list of data.
        """
        self.datas = datas

    @staticmethod
    def count(n: int, m: int) -> int:
        """
        Calculate the number of combinations for a specific count.
        :param n: The total number of elements,int.
        :param m: The number of elements in each combination,int.
        :return: The number of combinations,int.
        >>> CombinationCalculator.count(4, 2)
        6
        """

    @staticmethod
    def count_all(n: int) -> int:
        """
        Calculate the number of all possible combinations.
        :param n: The total number of elements,int.
        :return: The number of all possible combinations,int,if the number of combinations is greater than 2^63-1,return float("inf").
        >>> CombinationCalculator.count_all(4)
        15
        """

    def select(self, m: int) -> List[List[str]]:
        """
        Generate combinations with a specified number of elements.
        :param m: The number of elements in each combination,int.
        :return: A list of combinations,List[List[str]].
        >>> calc = CombinationCalculator(["A", "B", "C", "D"])
        >>> calc.select(2)
        [['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D']]

        """

    def select_all(self) -> List[List[str]]:
        """
        Generate all possible combinations of  selecting elements from the given data list,and it uses the select method.
        :return: A list of combinations,List[List[str]].
        >>> calc = CombinationCalculator(["A", "B", "C", "D"])
        >>> calc.select_all()
        [['A'], ['B'], ['C'], ['D'], ['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D'], ['A', 'B', 'C'], ['A', 'B', 'D'], ['A', 'C', 'D'], ['B', 'C', 'D'], ['A', 'B', 'C', 'D']]

        """

    def _select(self, dataIndex: int, resultList: List[str], resultIndex: int, result: List[List[str]]):
        """
        Generate combinations with a specified number of elements by recursion.
        :param dataIndex: The index of the data to be selected,int.
        :param resultList: The list of elements in the combination,List[str].
        :param resultIndex: The index of the element in the combination,int.
        :param result: The list of combinations,List[List[str]].
        :return: None.
        >>> calc = CombinationCalculator(["A", "B", "C", "D"])
        >>> result = []
        >>> calc._select(0, [None] * 2, 0, result)
        >>> result
        [['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D']]

        """
'''

import math
from typing import List

class CombinationCalculator:
    def __init__(self, datas: List[str]):
        self.datas = datas

    @staticmethod
    @inspect_code
    def count(n: int, m: int) -> int:
        if m == 0 or n == m:
            return 1
        return math.factorial(n) // (math.factorial(n - m) * math.factorial(m))

    @staticmethod
    @inspect_code
    def count_all(n: int) -> int:
        if n < 0 or n > 63:
            return False
        return (1 << n) - 1 if n != 63 else float("inf")

    @inspect_code
    def select(self, m: int) -> List[List[str]]:
        result = []
        self._select(0, [None] * m, 0, result)
        return result

    @inspect_code
    def select_all(self) -> List[List[str]]:
        result = []
        for i in range(1, len(self.datas) + 1):
            result.extend(self.select(i))
        return result

    @inspect_code
    def _select(self, dataIndex: int, resultList: List[str], resultIndex: int, result: List[List[str]]):
        resultLen = len(resultList)
        resultCount = resultIndex + 1
        if resultCount > resultLen:
            result.append(resultList.copy())
            return

        for i in range(dataIndex, len(self.datas) + resultCount - resultLen):
            resultList[resultIndex] = self.datas[i]
            self._select(i + 1, resultList, resultIndex + 1, result)


import unittest

class CombinationCalculatorTestCount(unittest.TestCase):
    def test_count(self):
        self.assertEqual(CombinationCalculator.count(4, 2), 6)
    def test_count_2(self):
        self.assertEqual(CombinationCalculator.count(5, 3), 10)

    def test_count_3(self):
        self.assertEqual(CombinationCalculator.count(6, 6), 1)

    def test_count_4(self):
        self.assertEqual(CombinationCalculator.count(6, 0), 1)

    def test_count_5(self):
        self.assertEqual(CombinationCalculator.count(6, 3), 20)

class CombinationCalculatorTestCountAll(unittest.TestCase):
    def test_count_all(self):
        self.assertEqual(CombinationCalculator.count_all(4), 15)

    def test_count_all_2(self):
        self.assertEqual(CombinationCalculator.count_all(-1), False)

    def test_count_all_3(self):
        self.assertEqual(CombinationCalculator.count_all(65), False)

    def test_count_all_4(self):
        self.assertEqual(CombinationCalculator.count_all(0), 0)

    def test_count_all_5(self):
        self.assertEqual(CombinationCalculator.count_all(63), float("inf"))

class CombinationCalculatorTestSelect(unittest.TestCase):
    def test_select(self):
        calc = CombinationCalculator(["A", "B", "C", "D"])
        self.assertEqual(calc.count(4, 2), 6)

    def test_select_2(self):
        calc = CombinationCalculator(["A", "B", "C", "D"])
        self.assertEqual(calc.count(5, 3), 10)

    def test_select_3(self):
        calc = CombinationCalculator(["A", "B", "C", "D"])
        self.assertEqual(calc.count(6, 6), 1)

    def test_select_4(self):
        calc = CombinationCalculator(["A", "B", "C", "D"])
        self.assertEqual(calc.count(6, 0), 1)

    def test_select_5(self):
        calc = CombinationCalculator(["A", "B", "C", "D"])
        self.assertEqual(calc.count(6, 3), 20)

class CombinationCalculatorTestSelectAll(unittest.TestCase):
    def test_select_all(self):
        calc = CombinationCalculator(["A"])
        self.assertEqual(calc.select_all(), [['A']])

    def test_select_all_2(self):
        calc = CombinationCalculator(["A", "B"])
        self.assertEqual(calc.select_all(), [['A'], ['B'], ['A', 'B']])

    def test_select_all_3(self):
        calc = CombinationCalculator(["A", "B", "C"])
        self.assertEqual(calc.select_all(),[['A'], ['B'], ['C'], ['A', 'B'], ['A', 'C'], ['B', 'C'], ['A', 'B', 'C']])

    def test_select_all_4(self):
        calc = CombinationCalculator([])
        self.assertEqual(calc.select_all(),[])

    def test_select_all_5(self):
        calc = CombinationCalculator(["B"])
        self.assertEqual(calc.select_all(),[['B']])


class CombinationCalculatorTestSelect2(unittest.TestCase):
    def test_select2(self):
        calc = CombinationCalculator(["A", "B", "C", "D"])
        result = []
        calc._select(0, [None] * 2, 0, result)
        self.assertEqual(result, [['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D']])

    def test_select2_2(self):
        calc = CombinationCalculator(["A", "B", "C", "D"])
        result = []
        calc._select(0, [None] * 3, 0, result)
        self.assertEqual(result, [['A', 'B', 'C'], ['A', 'B', 'D'], ['A', 'C', 'D'], ['B', 'C', 'D']])

    def test_select2_3(self):
        calc = CombinationCalculator(["A", "B", "C", "D"])
        result = []
        calc._select(0, [None] * 1, 0, result)
        self.assertEqual(result, [['A'], ['B'], ['C'], ['D']])

    def test_select2_4(self):
        calc = CombinationCalculator(["A", "B", "C", "D"])
        result = []
        calc._select(0, [None] * 0, 0, result)
        self.assertEqual(result, [[]])

    def test_select2_5(self):
        calc = CombinationCalculator(["A", "B", "C", "D"])
        result = []
        calc._select(0, [None] * 4, 0, result)
        self.assertEqual(result, [['A', 'B', 'C', 'D']])

class CombinationCalculatorTestMain(unittest.TestCase):
    def test_main(self):
        calc = CombinationCalculator(["A", "B", "C", "D"])
        self.assertEqual(calc.count(4, 2), 6)
        self.assertEqual(calc.count_all(4), 15)
        self.assertEqual(calc.select(2), [['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D']])
        self.assertEqual(calc.select_all(), [['A'], ['B'], ['C'], ['D'], ['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D'], ['A', 'B', 'C'], ['A', 'B', 'D'], ['A', 'C', 'D'], ['B', 'C', 'D'], ['A', 'B', 'C', 'D']])
        result = []
        calc._select(0, [None] * 2, 0, result)
        self.assertEqual(result, [['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D']])
