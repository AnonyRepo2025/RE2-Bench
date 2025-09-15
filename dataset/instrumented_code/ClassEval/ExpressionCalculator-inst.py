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
       jsonl_path = json_base + "/ExpressionCalculator.jsonl"
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
# This is a class in Python that can perform calculations with basic arithmetic operations, including addition, subtraction, multiplication, division, and modulo.

class ExpressionCalculator:
    def __init__(self):
        """
        Initialize the expression calculator
        """
        self.postfix_stack = deque()
        self.operat_priority = [0, 3, 2, 1, -1, 1, 0, 2]

    def calculate(self, expression):
        """
        Calculate the result of the given postfix expression
        :param expression: string, the postfix expression to be calculated
        :return: float, the calculated result
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.calculate("2 + 3 * 4")
        14.0

        """


    def prepare(self, expression):
        """
        Prepare the infix expression for conversion to postfix notation
        :param expression: string, the infix expression to be prepared
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.prepare("2+3*4")

        expression_calculator.postfix_stack = ['2', '3', '4', '*', '+']
        """


    @staticmethod
    def is_operator(c):
        """
        Check if a character is an operator in {'+', '-', '*', '/', '(', ')', '%'}
        :param c: string, the character to be checked
        :return: bool, True if the character is an operator, False otherwise
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.is_operator("+")
        True

        """


    def compare(self, cur, peek):
        """
        Compare the precedence of two operators
        :param cur: string, the current operator
        :param peek: string, the operator at the top of the operator stack
        :return: bool, True if the current operator has higher or equal precedence, False otherwise
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.compare("+", "-")
        True

        """


    @staticmethod
    def _calculate(first_value, second_value, current_op):
        """
        Perform the mathematical calculation based on the given operands and operator
        :param first_value: string, the first operand
        :param second_value: string, the second operand
        :param current_op: string, the operator
        :return: decimal.Decimal, the calculated result
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator._calculate("2", "3", "+")
        5.0

        """


    @staticmethod
    def transform(expression):
        """
        Transform the infix expression to a format suitable for conversion
        :param expression: string, the infix expression to be transformed
        :return: string, the transformed expression
        >>> expression_calculator = ExpressionCalculator()
        >>> expression_calculator.transform("2 + 3 * 4")
        "2+3*4"

        """

'''

import re
from collections import deque
from decimal import Decimal


class ExpressionCalculator:
    def __init__(self):
        self.postfix_stack = deque()
        self.operat_priority = [0, 3, 2, 1, -1, 1, 0, 2]

    @inspect_code
    def calculate(self, expression):
        self.prepare(self.transform(expression))

        result_stack = deque()
        self.postfix_stack.reverse()

        while self.postfix_stack:
            current_op = self.postfix_stack.pop()
            if not self.is_operator(current_op):
                current_op = current_op.replace("~", "-")
                result_stack.append(current_op)
            else:
                second_value = result_stack.pop()
                first_value = result_stack.pop()

                first_value = first_value.replace("~", "-")
                second_value = second_value.replace("~", "-")

                temp_result = self._calculate(first_value, second_value, current_op)
                result_stack.append(str(temp_result))

        return float(eval("*".join(result_stack)))

    @inspect_code
    def prepare(self, expression):
        op_stack = deque([','])
        arr = list(expression)
        current_index = 0
        count = 0

        for i, current_op in enumerate(arr):
            if self.is_operator(current_op):
                if count > 0:
                    self.postfix_stack.append("".join(arr[current_index: current_index + count]))
                peek_op = op_stack[-1]
                if current_op == ')':
                    while op_stack[-1] != '(':
                        self.postfix_stack.append(str(op_stack.pop()))
                    op_stack.pop()
                else:
                    while current_op != '(' and peek_op != ',' and self.compare(current_op, peek_op):
                        self.postfix_stack.append(str(op_stack.pop()))
                        peek_op = op_stack[-1]
                    op_stack.append(current_op)

                count = 0
                current_index = i + 1
            else:
                count += 1

        if count > 1 or (count == 1 and not self.is_operator(arr[current_index])):
            self.postfix_stack.append("".join(arr[current_index: current_index + count]))

        while op_stack[-1] != ',':
            self.postfix_stack.append(str(op_stack.pop()))

    @staticmethod
    @inspect_code
    def is_operator(c):
        return c in {'+', '-', '*', '/', '(', ')', '%'}

    @inspect_code
    def compare(self, cur, peek):
        if cur == '%':
            cur = '/'
        if peek == '%':
            peek = '/'
        return self.operat_priority[ord(peek) - 40] >= self.operat_priority[ord(cur) - 40]

    @staticmethod
    @inspect_code
    def _calculate(first_value, second_value, current_op):
        if current_op == '+':
            return Decimal(first_value) + Decimal(second_value)
        elif current_op == '-':
            return Decimal(first_value) - Decimal(second_value)
        elif current_op == '*':
            return Decimal(first_value) * Decimal(second_value)
        elif current_op == '/':
            return Decimal(first_value) / Decimal(second_value)
        elif current_op == '%':
            return Decimal(first_value) % Decimal(second_value)
        else:
            raise ValueError("Unexpected operator: {}".format(current_op))

    @staticmethod
    @inspect_code
    def transform(expression):
        expression = re.sub(r"\s+", "", expression)
        expression = re.sub(r"=$", "", expression)
        arr = list(expression)

        for i, c in enumerate(arr):
            if c == '-':
                if i == 0:
                    arr[i] = '~'
                else:
                    prev_c = arr[i - 1]
                    if prev_c in {'+', '-', '*', '/', '(', 'E', 'e'}:
                        arr[i] = '~'

        if arr[0] == '~' and (len(arr) > 1 and arr[1] == '('):
            arr[0] = '-'
            return "0" + "".join(arr)
        else:
            return "".join(arr)



import unittest


class ExpressionCalculatorTestCalculate(unittest.TestCase):
    def setUp(self):
        self.expression_calculator = ExpressionCalculator()

    def test_calculate_1(self):
        result = self.expression_calculator.calculate("2 + 3 * 4")
        self.assertEqual(result, 14.0)

    def test_calculate_2(self):
        result = self.expression_calculator.calculate("2 + 3 + 4")
        self.assertEqual(result, 9.0)

    def test_calculate_3(self):
        result = self.expression_calculator.calculate("2 * 3 * 4")
        self.assertEqual(result, 24.0)

    def test_calculate_4(self):
        result = self.expression_calculator.calculate("2 + 4 / 4")
        self.assertEqual(result, 3.0)

    def test_calculate_5(self):
        result = self.expression_calculator.calculate("(2 + 3) * 4")
        self.assertEqual(result, 20.0)


class ExpressionCalculatorTestPrepare(unittest.TestCase):
    def setUp(self):
        self.expression_calculator = ExpressionCalculator()

    def test_prepare_1(self):
        self.expression_calculator.prepare("2+3*4")
        self.assertEqual(self.expression_calculator.postfix_stack, deque(['2', '3', '4', '*', '+']))

    def test_prepare_2(self):
        self.expression_calculator.prepare("2+3/4")
        self.assertEqual(self.expression_calculator.postfix_stack, deque(['2', '3', '4', '/', '+']))

    def test_prepare_3(self):
        self.expression_calculator.prepare("2-3*4")
        self.assertEqual(self.expression_calculator.postfix_stack, deque(['2', '3', '4', '*', '-']))

    def test_prepare_4(self):
        self.expression_calculator.prepare("1+3*4")
        self.assertEqual(self.expression_calculator.postfix_stack, deque(['1', '3', '4', '*', '+']))

    def test_prepare_5(self):
        self.expression_calculator.prepare("(2+3)*4")
        self.assertEqual(self.expression_calculator.postfix_stack, deque(['2', '3', '+', '4', '*']))

    def test_prepare_6(self):
        self.expression_calculator.prepare("")
        self.assertEqual(self.expression_calculator.postfix_stack, deque([]))


class ExpressionCalculatorTestIsOperator(unittest.TestCase):
    def setUp(self):
        self.expression_calculator = ExpressionCalculator()

    def test_is_operator_1(self):
        self.assertTrue(self.expression_calculator.is_operator("+"))

    def test_is_operator_2(self):
        self.assertTrue(self.expression_calculator.is_operator("-"))

    def test_is_operator_3(self):
        self.assertTrue(self.expression_calculator.is_operator("*"))

    def test_is_operator_4(self):
        self.assertTrue(self.expression_calculator.is_operator("/"))

    def test_is_operator_5(self):
        self.assertFalse(self.expression_calculator.is_operator("5"))


class ExpressionCalculatorTestCompare(unittest.TestCase):
    def setUp(self):
        self.expression_calculator = ExpressionCalculator()

    def test_compare_1(self):
        result = self.expression_calculator.compare("+", "-")
        self.assertTrue(result)

    def test_compare_2(self):
        result = self.expression_calculator.compare("*", "/")
        self.assertTrue(result)

    def test_compare_3(self):
        result = self.expression_calculator.compare("+", "*")
        self.assertTrue(result)

    def test_compare_4(self):
        result = self.expression_calculator.compare("*", "+")
        self.assertFalse(result)

    def test_compare_5(self):
        result = self.expression_calculator.compare("/", "+")
        self.assertFalse(result)

    def test_compare_6(self):
        result = self.expression_calculator.compare("%", "+")
        self.assertFalse(result)

    def test_compare_7(self):
        result = self.expression_calculator.compare("+", "%")
        self.assertTrue(result)


class ExpressionCalculatorTestCalculateMethod(unittest.TestCase):
    def setUp(self):
        self.expression_calculator = ExpressionCalculator()

    def test_calculate_method_1(self):
        result = self.expression_calculator._calculate("2", "3", "+")
        self.assertEqual(result, Decimal(5.0))

    def test_calculate_method_2(self):
        result = self.expression_calculator._calculate("3", "2", "-")
        self.assertEqual(result, Decimal(1.0))

    def test_calculate_method_3(self):
        result = self.expression_calculator._calculate("2", "3", "*")
        self.assertEqual(result, Decimal(6.0))

    def test_calculate_method_4(self):
        result = self.expression_calculator._calculate("3", "3", "/")
        self.assertEqual(result, Decimal(1.0))

    def test_calculate_method_5(self):
        result = self.expression_calculator._calculate("6", "2", "/")
        self.assertEqual(result, Decimal(3.0))

    def test_calculate_method_6(self):
        result = self.expression_calculator._calculate("6", "2", "%")
        self.assertEqual(result, Decimal(0.0))

    def test_calculate_method_7(self):
        try:
            self.expression_calculator._calculate("6", "2", "??")
        except:
            pass


class ExpressionCalculatorTestTransform(unittest.TestCase):
    def setUp(self):
        self.expression_calculator = ExpressionCalculator()

    def test_transform_1(self):
        result = self.expression_calculator.transform("2 + 3 * 4")
        self.assertEqual(result, "2+3*4")

    def test_transform_2(self):
        result = self.expression_calculator.transform("2 + 3 / 4")
        self.assertEqual(result, "2+3/4")

    def test_transform_3(self):
        result = self.expression_calculator.transform("2 - 3 * 4")
        self.assertEqual(result, "2-3*4")

    def test_transform_4(self):
        result = self.expression_calculator.transform("1 + 3 * 4")
        self.assertEqual(result, "1+3*4")

    def test_transform_5(self):
        result = self.expression_calculator.transform("-2 + (-3) * 4")
        self.assertEqual(result, "~2+(~3)*4")

    def test_transform_6(self):
        result = self.expression_calculator.transform("~(1 + 1)")
        self.assertEqual(result, "0-(1+1)")


class ExpressionCalculatorTest(unittest.TestCase):
    def setUp(self):
        self.expression_calculator = ExpressionCalculator()

    def test_ExpressionCalculator(self):
        result = self.expression_calculator.calculate("2 + 3 * 4")
        self.assertEqual(result, 14.0)

        self.expression_calculator.prepare("2+3*4")
        self.assertEqual(self.expression_calculator.postfix_stack, deque(['2', '3', '4', '*', '+']))

        self.assertTrue(self.expression_calculator.is_operator("+"))

        result = self.expression_calculator.compare("+", "-")
        self.assertTrue(result)

        result = self.expression_calculator._calculate("2", "3", "+")
        self.assertEqual(result, Decimal(5.0))

        result = self.expression_calculator.transform("2 + 3 * 4")
        self.assertEqual(result, "2+3*4")

