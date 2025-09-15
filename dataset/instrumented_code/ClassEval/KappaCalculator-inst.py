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
       jsonl_path = json_base + "/KappaCalculator.jsonl"
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
# This is a class as KappaCalculator, supporting to calculate Cohen's and Fleiss' kappa coefficient.

import numpy as np

class KappaCalculator:

    @staticmethod
    def kappa(testData, k):
        """
        Calculate the cohens kappa value of a k-dimensional matrix
        :param testData: The k-dimensional matrix that needs to calculate the cohens kappa value
        :param k: int, Matrix dimension
        :return:float, the cohens kappa value of the matrix
        >>> KappaCalculator.kappa([[2, 1, 1], [1, 2, 1], [1, 1, 2]], 3)
        0.25
        """

    @staticmethod
    def fleiss_kappa(testData, N, k, n):
        """
        Calculate the fliss kappa value of an N * k matrix
        :param testData: Input data matrix, N * k
        :param N: int, Number of samples
        :param k: int, Number of categories
        :param n: int, Number of raters
        :return: float, fleiss kappa value
        >>> KappaCalculator.fleiss_kappa([[0, 0, 0, 0, 14],
        >>>                              [0, 2, 6, 4, 2],
        >>>                              [0, 0, 3, 5, 6],
        >>>                              [0, 3, 9, 2, 0],
        >>>                              [2, 2, 8, 1, 1],
        >>>                              [7, 7, 0, 0, 0],
        >>>                              [3, 2, 6, 3, 0],
        >>>                              [2, 5, 3, 2, 2],
        >>>                              [6, 5, 2, 1, 0],
        >>>                              [0, 2, 2, 3, 7]], 10, 5, 14)
        0.20993070442195522
        """
'''

import numpy as np


class KappaCalculator:

    @staticmethod
    @inspect_code
    def kappa(testData, k):
        dataMat = np.mat(testData)
        P0 = 0.0
        for i in range(k):
            P0 += dataMat[i, i] * 1.0
        xsum = np.sum(dataMat, axis=1)
        ysum = np.sum(dataMat, axis=0)
        sum = np.sum(dataMat)
        Pe = float(ysum * xsum) / sum / sum
        P0 = float(P0 / sum * 1.0)
        cohens_coefficient = float((P0 - Pe) / (1 - Pe))
        return cohens_coefficient

    @staticmethod
    @inspect_code
    def fleiss_kappa(testData, N, k, n):
        dataMat = np.mat(testData, float)
        oneMat = np.ones((k, 1))
        sum = 0.0
        P0 = 0.0
        for i in range(N):
            temp = 0.0
            for j in range(k):
                sum += dataMat[i, j]
                temp += 1.0 * dataMat[i, j] ** 2
            temp -= n
            temp /= (n - 1) * n
            P0 += temp
        P0 = 1.0 * P0 / N
        ysum = np.sum(dataMat, axis=0)
        for i in range(k):
            ysum[0, i] = (ysum[0, i] / sum) ** 2
        Pe = ysum * oneMat * 1.0
        ans = (P0 - Pe) / (1 - Pe)
        return ans[0, 0]

import unittest


class KappaCalculatorTestKappa(unittest.TestCase):
    def test_kappa_1(self):
        self.assertEqual(KappaCalculator.kappa([[2, 1, 1], [1, 2, 1], [1, 1, 2]], 3), 0.25)

    def test_kappa_2(self):
        self.assertAlmostEqual(KappaCalculator.kappa([[2, 2, 1], [1, 2, 1], [1, 1, 2]], 3), 0.19469026548672572)

    def test_kappa_3(self):
        self.assertAlmostEqual(KappaCalculator.kappa([[2, 1, 2], [1, 2, 1], [1, 1, 2]], 3), 0.19469026548672572)

    def test_kappa_4(self):
        self.assertAlmostEqual(KappaCalculator.kappa([[2, 1, 1], [2, 2, 1], [1, 1, 2]], 3), 0.19469026548672572)

    def test_kappa_5(self):
        self.assertAlmostEqual(KappaCalculator.kappa([[2, 1, 1], [1, 2, 2], [1, 1, 2]], 3), 0.19469026548672572)


class KappaCalculatorTestFleissKappa(unittest.TestCase):
    def test_fleiss_kappa_1(self):
        self.assertEqual(KappaCalculator.fleiss_kappa([[0, 0, 0, 0, 14],
                                                       [0, 2, 6, 4, 2],
                                                       [0, 0, 3, 5, 6],
                                                       [0, 3, 9, 2, 0],
                                                       [2, 2, 8, 1, 1],
                                                       [7, 7, 0, 0, 0],
                                                       [3, 2, 6, 3, 0],
                                                       [2, 5, 3, 2, 2],
                                                       [6, 5, 2, 1, 0],
                                                       [0, 2, 2, 3, 7]], 10, 5, 14), 0.20993070442195522)

    def test_fleiss_kappa_2(self):
        self.assertEqual(KappaCalculator.fleiss_kappa([[1, 0, 0, 0, 14],
                                                       [0, 2, 6, 4, 2],
                                                       [0, 0, 3, 5, 6],
                                                       [0, 3, 9, 2, 0],
                                                       [2, 2, 8, 1, 1],
                                                       [7, 7, 0, 0, 0],
                                                       [3, 2, 6, 3, 0],
                                                       [2, 5, 3, 2, 2],
                                                       [6, 5, 2, 1, 0],
                                                       [0, 2, 2, 3, 7]], 10, 5, 14), 0.2115748928799344)

    def test_fleiss_kappa_3(self):
        self.assertEqual(KappaCalculator.fleiss_kappa([[0, 1, 0, 0, 14],
                                                       [0, 2, 6, 4, 2],
                                                       [0, 0, 3, 5, 6],
                                                       [0, 3, 9, 2, 0],
                                                       [2, 2, 8, 1, 1],
                                                       [7, 7, 0, 0, 0],
                                                       [3, 2, 6, 3, 0],
                                                       [2, 5, 3, 2, 2],
                                                       [6, 5, 2, 1, 0],
                                                       [0, 2, 2, 3, 7]], 10, 5, 14), 0.21076904123090398)

    def test_fleiss_kappa_4(self):
        self.assertEqual(KappaCalculator.fleiss_kappa([[0, 0, 1, 0, 14],
                                                       [0, 2, 6, 4, 2],
                                                       [0, 0, 3, 5, 6],
                                                       [0, 3, 9, 2, 0],
                                                       [2, 2, 8, 1, 1],
                                                       [7, 7, 0, 0, 0],
                                                       [3, 2, 6, 3, 0],
                                                       [2, 5, 3, 2, 2],
                                                       [6, 5, 2, 1, 0],
                                                       [0, 2, 2, 3, 7]], 10, 5, 14), 0.2096583016522883)

    def test_fleiss_kappa_5(self):
        self.assertEqual(KappaCalculator.fleiss_kappa([[0, 0, 0, 1, 14],
                                                       [0, 2, 6, 4, 2],
                                                       [0, 0, 3, 5, 6],
                                                       [0, 3, 9, 2, 0],
                                                       [2, 2, 8, 1, 1],
                                                       [7, 7, 0, 0, 0],
                                                       [3, 2, 6, 3, 0],
                                                       [2, 5, 3, 2, 2],
                                                       [6, 5, 2, 1, 0],
                                                       [0, 2, 2, 3, 7]], 10, 5, 14), 0.21147425143148907)


class KappaCalculatorTest(unittest.TestCase):
    def test_kappacalculator(self):
        self.assertEqual(KappaCalculator.kappa([[2, 1, 1], [1, 2, 1], [1, 1, 2]], 3), 0.25)
        self.assertEqual(KappaCalculator.fleiss_kappa([[0, 0, 0, 0, 14],
                                                       [0, 2, 6, 4, 2],
                                                       [0, 0, 3, 5, 6],
                                                       [0, 3, 9, 2, 0],
                                                       [2, 2, 8, 1, 1],
                                                       [7, 7, 0, 0, 0],
                                                       [3, 2, 6, 3, 0],
                                                       [2, 5, 3, 2, 2],
                                                       [6, 5, 2, 1, 0],
                                                       [0, 2, 2, 3, 7]], 10, 5, 14), 0.20993070442195522)

