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
       jsonl_path = json_base + "/SignInSystem.jsonl"
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
# This is a class as sigin in system, including adding users, signing in/out, checking sign-in status, and retrieving signed-in/not signed-in users.

class SignInSystem:
    def __init__(self):
        """
        Initialize the sign-in system.
        """
        self.users = {}

    def add_user(self, username):
        """
        Add a user to the sign-in system if the user wasn't in the self.users.
        And the initial state is False.
        :param username: str, the username to be added.
        :return: bool, True if the user is added successfully, False if the user already exists.
        >>> signInSystem.add_user("mike")
        True
        >>> signInSystem.add_user("mike")
        False
        """

    def sign_in(self, username):
        """
        Sign in a user if the user was in the self.users and change the state to True.
        :param username: str, the username to be signed in.
        :return: bool, True if the user is signed in successfully, False if the user does not exist.
        >>> signInSystem.sign_in("mike")
        True
        >>> signInSystem.sign_in("mik")
        False
        """

    def check_sign_in(self, username):
        """
        Check if a user is signed in.
        :param username: str, the username to be checked.
        :return: bool, True if the user is signed in, False if the user does not exist or is not signed in.
        >>> signInSystem.check_sign_in("jack")
        False
        >>> signInSystem.add_user("jack")
        >>> signInSystem.check_sign_in("jack")
        >>> signInSystem.sign_in("jack")
        >>> signInSystem.check_sign_in("jack")
        True
        """

    def all_signed_in(self):
        """
        Check if all users are signed in.
        :return: bool, True if all users are signed in, False otherwise.
        >>> signInSystem.add_user("jack")
        True
        >>> signInSystem.sign_in("jack")
        >>> signInSystem.all_signed_in()
        True
        """

    def all_not_signed_in(self):
        """
        Get a list of usernames that are not signed in.
        :return: list[str], a list of usernames that are not signed in.
        >>> signInSystem = SignInSystem()
        >>> signInSystem.add_user("a")
        True
        >>> signInSystem.add_user("b")
        True
        >>> signInSystem.all_not_signed_in()
        ['a', 'b']
        """
'''


class SignInSystem:
    def __init__(self):
        self.users = {}

    @inspect_code
    def add_user(self, username):
        if username in self.users:
            return False
        else:
            self.users[username] = False
            return True

    @inspect_code
    def sign_in(self, username):
        if username not in self.users:
            return False
        else:
            self.users[username] = True
            return True

    @inspect_code
    def check_sign_in(self, username):
        if username not in self.users:
            return False
        else:
            if self.users[username]:
                return True
            else:
                return False

    @inspect_code
    def all_signed_in(self):
        if all(self.users.values()):
            return True
        else:
            return False

    @inspect_code
    def all_not_signed_in(self):
        not_signed_in_users = []
        for username, signed_in in self.users.items():
            if not signed_in:
                not_signed_in_users.append(username)
        return not_signed_in_users

import unittest


class SignInSystemTestAddUser(unittest.TestCase):
    def test_add_user_1(self):
        signin_system = SignInSystem()
        result = signin_system.add_user("user1")
        self.assertTrue(result)

    def test_add_user_2(self):
        signin_system = SignInSystem()
        signin_system.add_user("user1")
        result = signin_system.add_user("user1")
        self.assertFalse(result)

    def test_add_user_3(self):
        signin_system = SignInSystem()
        result = signin_system.add_user("aaa")
        self.assertTrue(result)

    def test_add_user_4(self):
        signin_system = SignInSystem()
        result = signin_system.add_user("bbb")
        self.assertTrue(result)

    def test_add_user_5(self):
        signin_system = SignInSystem()
        result = signin_system.add_user("ccc")
        self.assertTrue(result)


class SignInSystemTestSignIn(unittest.TestCase):
    def test_sign_in_1(self):
        signin_system = SignInSystem()
        signin_system.add_user("user1")
        result = signin_system.sign_in("user1")
        self.assertTrue(result)

    # user not exist
    def test_sign_in_2(self):
        signin_system = SignInSystem()
        result = signin_system.sign_in("user1")
        self.assertFalse(result)

    def test_sign_in_3(self):
        signin_system = SignInSystem()
        signin_system.add_user("aaa")
        result = signin_system.sign_in("aaa")
        self.assertTrue(result)

    def test_sign_in_4(self):
        signin_system = SignInSystem()
        signin_system.add_user("bbb")
        result = signin_system.sign_in("bbb")
        self.assertTrue(result)

    def test_sign_in_5(self):
        signin_system = SignInSystem()
        result = signin_system.sign_in("ccc")
        self.assertFalse(result)


class SignInSystemTestCheckSignIn(unittest.TestCase):
    # has signed in
    def test_check_sign_in_1(self):
        signin_system = SignInSystem()
        signin_system.add_user("user1")
        signin_system.sign_in("user1")
        result = signin_system.check_sign_in("user1")
        self.assertTrue(result)

    # hasn't signed in 
    def test_check_sign_in_2(self):
        signin_system = SignInSystem()
        signin_system.add_user("user1")
        result = signin_system.check_sign_in("user1")
        self.assertFalse(result)

    # not exist
    def test_check_sign_in_3(self):
        signin_system = SignInSystem()
        result = signin_system.check_sign_in("user1")
        self.assertFalse(result)

    def test_check_sign_in_4(self):
        signin_system = SignInSystem()
        signin_system.add_user("aaa")
        signin_system.sign_in("aaa")
        result = signin_system.check_sign_in("aaa")
        self.assertTrue(result)

    def test_check_sign_in_5(self):
        signin_system = SignInSystem()
        signin_system.add_user("bbb")
        signin_system.sign_in("bbb")
        result = signin_system.check_sign_in("bbb")
        self.assertTrue(result)


class SignInSystemTestAllSignedIn(unittest.TestCase):
    def test_all_signed_in_1(self):
        signin_system = SignInSystem()
        signin_system.add_user("user1")
        signin_system.sign_in("user1")
        result = signin_system.all_signed_in()
        self.assertTrue(result)

    def test_all_signed_in_2(self):
        signin_system = SignInSystem()
        signin_system.add_user("user1")
        result = signin_system.all_signed_in()
        self.assertFalse(result)

    def test_all_signed_in_3(self):
        signin_system = SignInSystem()
        signin_system.add_user("aaa")
        signin_system.sign_in("aaa")
        result = signin_system.all_signed_in()
        self.assertTrue(result)

    def test_all_signed_in_4(self):
        signin_system = SignInSystem()
        signin_system.add_user("bbb")
        signin_system.sign_in("bbb")
        result = signin_system.all_signed_in()
        self.assertTrue(result)

    def test_all_signed_in_5(self):
        signin_system = SignInSystem()
        signin_system.add_user("aaa")
        signin_system.add_user("bbb")
        signin_system.sign_in("aaa")
        result = signin_system.all_signed_in()
        self.assertFalse(result)


class SignInSystemTestAllNotSignedIn(unittest.TestCase):
    def test_all_not_signed_in_1(self):
        signin_system = SignInSystem()
        signin_system.add_user("user1")
        signin_system.sign_in("user1")
        result = signin_system.all_not_signed_in()
        self.assertEqual([], result)

    def test_all_not_signed_in_2(self):
        signin_system = SignInSystem()
        signin_system.add_user("user1")
        signin_system.add_user("user2")
        result = signin_system.all_not_signed_in()
        self.assertEqual(["user1", "user2"], result)

    def test_all_not_signed_in_3(self):
        signin_system = SignInSystem()
        signin_system.add_user("aaa")
        signin_system.sign_in("aaa")
        result = signin_system.all_not_signed_in()
        self.assertEqual([], result)

    def test_all_not_signed_in_4(self):
        signin_system = SignInSystem()
        signin_system.add_user("user1")
        signin_system.add_user("aaa")
        signin_system.sign_in("user1")
        result = signin_system.all_not_signed_in()
        self.assertEqual(['aaa'], result)

    def test_all_not_signed_in_5(self):
        signin_system = SignInSystem()
        result = signin_system.all_not_signed_in()
        self.assertEqual([], result)


class SignInSystemTestMain(unittest.TestCase):
    def setUp(self):
        self.signin_system = SignInSystem()

    def test_main(self):
        result = self.signin_system.add_user("user1")
        result = self.signin_system.add_user("user2")
        self.assertTrue(result)

        result = self.signin_system.sign_in("user1")
        self.assertTrue(result)

        result = self.signin_system.check_sign_in("user1")
        self.assertTrue(result)

        result = self.signin_system.all_signed_in()
        self.assertFalse(result)

        result = self.signin_system.all_not_signed_in()
        self.assertEqual(["user2"], result)
