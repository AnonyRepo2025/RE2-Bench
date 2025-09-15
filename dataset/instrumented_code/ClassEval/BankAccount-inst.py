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
       jsonl_path = json_base + "/BankAccount.jsonl"
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
# This is a class as a bank account system, which supports deposit money, withdraw money, view balance, and transfer money.

class BankAccount:
    def __init__(self, balance=0):
        """
        Initializes a bank account object with an attribute balance, default value is 0.
        """
        self.balance = balance

    def deposit(self, amount):
        """
        Deposits a certain amount into the account, increasing the account balance, return the current account balance.
        If amount is negative, raise a ValueError("Invalid amount").
        :param amount: int
        """

    def withdraw(self, amount):
        """
        Withdraws a certain amount from the account, decreasing the account balance, return the current account balance.
        If amount is negative, raise a ValueError("Invalid amount").
        If the withdrawal amount is greater than the account balance, raise a ValueError("Insufficient balance.").
        :param amount: int
        """

    def view_balance(self):
        """
        Return the account balance.
        """

    def transfer(self, other_account, amount):
        """
        Transfers a certain amount from the current account to another account.
        :param other_account: BankAccount
        :param amount: int
        >>> account1 = BankAccount()
        >>> account2 = BankAccount()
        >>> account1.deposit(1000)
        >>> account1.transfer(account2, 300)
        account1.balance = 700 account2.balance = 300
        """
'''

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    @inspect_code
    def deposit(self, amount):
        if amount < 0:
            raise ValueError("Invalid amount")
        self.balance += amount
        return self.balance

    @inspect_code
    def withdraw(self, amount):
        if amount < 0:
            raise ValueError("Invalid amount")
        if amount > self.balance:
            raise ValueError("Insufficient balance.")
        self.balance -= amount
        return self.balance

    @inspect_code
    def view_balance(self):
        return self.balance

    @inspect_code
    def transfer(self, other_account, amount):
        self.withdraw(amount)
        other_account.deposit(amount)

import unittest

class BankAccountTestDeposit(unittest.TestCase):

    def test_deposit(self):
        account1 = BankAccount()
        ret = account1.deposit(1000)
        self.assertEqual(ret, 1000)

    def test_deposit_2(self):
        account1 = BankAccount()
        account1.deposit(1000)
        ret = account1.deposit(2000)
        self.assertEqual(ret, 3000)


    def test_deposit_3(self):
        account1 = BankAccount()
        with self.assertRaises(ValueError) as context:
            account1.deposit(-1000)
        self.assertEqual(str(context.exception), "Invalid amount")

    def test_deposit_4(self):
        account1 = BankAccount()
        ret = account1.deposit(0)
        self.assertEqual(ret, 0)

    def test_deposit_5(self):
        account1 = BankAccount()
        account1.deposit(1000)
        ret = account1.deposit(1000)
        self.assertEqual(ret, 2000)

class BankAccountTestWithdraw(unittest.TestCase):

    def test_withdraw(self):
        account1 = BankAccount()
        account1.balance = 1000
        ret = account1.withdraw(200)
        self.assertEqual(ret, 800)

    def test_withdraw_2(self):
        account1 = BankAccount()
        account1.balance = 500
        with self.assertRaises(ValueError) as context:
            account1.withdraw(1000)
        self.assertEqual(str(context.exception), "Insufficient balance.")

    def test_withdraw_3(self):
        account1 = BankAccount()
        with self.assertRaises(ValueError) as context:
            account1.withdraw(-1000)
        self.assertEqual(str(context.exception), "Invalid amount")

    def test_withdraw_4(self):
        account1 = BankAccount()
        account1.balance = 1000
        ret = account1.withdraw(500)
        self.assertEqual(ret, 500)

    def test_withdraw_5(self):
        account1 = BankAccount()
        account1.balance = 1000
        ret = account1.withdraw(1000)
        self.assertEqual(ret, 0)

class BankAccountTestViewBalance(unittest.TestCase):

    def test_view_balance(self):
        account1 = BankAccount()
        self.assertEqual(account1.view_balance(), 0)

    def test_view_balance_2(self):
        account1 = BankAccount()
        account1.balance = 1000
        self.assertEqual(account1.view_balance(), 1000)

    def test_view_balance_3(self):
        account1 = BankAccount()
        account1.balance = 500
        self.assertEqual(account1.view_balance(), 500)

    def test_view_balance_4(self):
        account1 = BankAccount()
        account1.balance = 1500
        self.assertEqual(account1.view_balance(), 1500)

    def test_view_balance_5(self):
        account1 = BankAccount()
        account1.balance = 2000
        self.assertEqual(account1.view_balance(), 2000)

class BankAccountTestTransfer(unittest.TestCase):

    def test_transfer(self):
        account1 = BankAccount()
        account2 = BankAccount()
        account1.balance = 800
        account2.balance = 1000
        account1.transfer(account2, 300)
        self.assertEqual(account1.view_balance(), 500)
        self.assertEqual(account2.view_balance(), 1300)

    def test_transfer_2(self):
        account1 = BankAccount()
        account2 = BankAccount()
        account1.balance = 500
        with self.assertRaises(ValueError) as context:
            account1.transfer(account2, 600)
        self.assertEqual(str(context.exception), "Insufficient balance.")

    def test_transfer_3(self):
        account1 = BankAccount()
        account2 = BankAccount()
        account1.balance = 500
        account2.balance = 1000
        with self.assertRaises(ValueError) as context:
            account1.transfer(account2, -600)
        self.assertEqual(str(context.exception), "Invalid amount")

    def test_transfer_4(self):
        account1 = BankAccount()
        account2 = BankAccount()
        account1.balance = 500
        account2.balance = 1000
        account1.transfer(account2, 500)
        self.assertEqual(account1.view_balance(), 0)
        self.assertEqual(account2.view_balance(), 1500)

    def test_transfer_5(self):
        account1 = BankAccount()
        account2 = BankAccount()
        account1.balance = 500
        account2.balance = 1000
        account1.transfer(account2, 200)
        self.assertEqual(account1.view_balance(), 300)
        self.assertEqual(account2.view_balance(), 1200)

class BankAccountTest(unittest.TestCase):

    def test_all(self):
        account1 = BankAccount()
        account2 = BankAccount()
        account1.deposit(1000)
        account1.withdraw(200)
        account1.transfer(account2, 300)
        self.assertEqual(account1.view_balance(), 500)
        self.assertEqual(account2.view_balance(), 300)

    def test_all2(self):
        account1 = BankAccount()
        account2 = BankAccount()
        account1.deposit(1000)
        account1.withdraw(200)
        account1.transfer(account2, 300)
        account2.withdraw(100)
        self.assertEqual(account1.view_balance(), 500)
        self.assertEqual(account2.view_balance(), 200)



