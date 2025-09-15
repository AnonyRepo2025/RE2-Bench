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
       jsonl_path = json_base + "/SQLGenerator.jsonl"
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
# This class generates SQL statements for common operations on a table, such as SELECT, INSERT, UPDATE, and DELETE.

class SQLGenerator:
    def __init__(self, table_name):
        """
        Initialize the table name.
        :param table_name: str
        """
        self.table_name = table_name

    def select(self, fields=None, condition=None):
        """
        Generates a SELECT SQL statement based on the specified fields and conditions.
        :param fields: list, optional. Default is None. The list of fields to be queried.
        :param condition: str, optional. Default is None. The condition expression for the query.
        :return: str. The generated SQL statement.
        >>> sql = SQLGenerator('table1')
        >>> sql.select(['field1', 'field2'], 'filed3 = value1')
        'SELECT field1, field2 FROM table1 WHERE filed3 = value1;'
        """

    def insert(self, data):
        """
        Generates an INSERT SQL statement based on the given data.
        :param data: dict. The data to be inserted, in dictionary form where keys are field names and values are field values.
        :return: str. The generated SQL statement.
        >>> sql.insert({'key1': 'value1', 'key2': 'value2'})
        "INSERT INTO table1 (key1, key2) VALUES ('value1', 'value2');"
        """


    def update(self, data, condition):
        """
        Generates an UPDATE SQL statement based on the given data and condition.
        :param data: dict. The data to be updated, in dictionary form where keys are field names and values are new field values.
        :param condition: str. The condition expression for the update.
        :return: str. The generated SQL statement.
        >>> sql.update({'field1': 'new_value1', 'field2': 'new_value2'}, "field3 = value1")
        "UPDATE table1 SET field1 = 'new_value1', field2 = 'new_value2' WHERE field3 = value1;"
        """

    def delete(self, condition):
        """
        Generates a DELETE SQL statement based on the given condition.
        :param condition: str. The condition expression for the delete.
        :return: str. The generated SQL statement.
        >>> sql.delete("field1 = value1")
        'DELETE FROM table1 WHERE field1 = value1;'
        """

    def select_female_under_age(self, age):
        """
        Generates a SQL statement to select females under a specified age.
        :param age: int. The specified age.
        :return: str. The generated SQL statement.
        >>> sql.select_female_under_age(30)
        "SELECT * FROM table1 WHERE age < 30 AND gender = 'female';"
        """

    def select_by_age_range(self, min_age, max_age):
        """
        Generates a SQL statement to select records within a specified age range.
        :param min_age: int. The minimum age.
        :param max_age: int. The maximum age.
        :return: str. The generated SQL statement.
        >>> sql.select_by_age_range(20, 30)
        'SELECT * FROM table1 WHERE age BETWEEN 20 AND 30;'
        """
'''


class SQLGenerator:
    def __init__(self, table_name):
        self.table_name = table_name

    @inspect_code
    def select(self, fields=None, condition=None):
        if fields is None:
            fields = "*"
        else:
            fields = ", ".join(fields)
        sql = f"SELECT {fields} FROM {self.table_name}"
        if condition is not None:
            sql += f" WHERE {condition}"
        return sql + ";"

    @inspect_code
    def insert(self, data):
        fields = ", ".join(data.keys())
        values = ", ".join([f"'{value}'" for value in data.values()])
        sql = f"INSERT INTO {self.table_name} ({fields}) VALUES ({values})"
        return sql + ";"

    @inspect_code
    def update(self, data, condition):
        set_clause = ", ".join([f"{field} = '{value}'" for field, value in data.items()])
        sql = f"UPDATE {self.table_name} SET {set_clause} WHERE {condition}"
        return sql + ";"

    @inspect_code
    def delete(self, condition):
        sql = f"DELETE FROM {self.table_name} WHERE {condition}"
        return sql + ";"

    @inspect_code
    def select_female_under_age(self, age):
        condition = f"age < {age} AND gender = 'female'"
        return self.select(condition=condition)

    @inspect_code
    def select_by_age_range(self, min_age, max_age):
        condition = f"age BETWEEN {min_age} AND {max_age}"
        return self.select(condition=condition)

import unittest

class SQLGeneratorTestSelect(unittest.TestCase):
    def test_select_1(self):
        sql = SQLGenerator('table1')
        result = sql.select(['field1'], "field2 = value1")
        self.assertEqual(result, "SELECT field1 FROM table1 WHERE field2 = value1;")

    def test_select_2(self):
        sql = SQLGenerator('table1')
        result = sql.select(['field1', 'field2'], "field3 = value1")
        self.assertEqual(result, "SELECT field1, field2 FROM table1 WHERE field3 = value1;")

    def test_select_3(self):
        sql = SQLGenerator('table1')
        result = sql.select(['field1, field2'], "field3 = value1")
        self.assertEqual(result, "SELECT field1, field2 FROM table1 WHERE field3 = value1;")

    def test_select_4(self):
        sql = SQLGenerator('table1')
        result = sql.select(['field1, field2'], "field3 = value1, field4 = value2")
        self.assertEqual(result, "SELECT field1, field2 FROM table1 WHERE field3 = value1, field4 = value2;")

    def test_select_5(self):
        sql = SQLGenerator('table1')
        result = sql.select(['field1'], "field2 = value1, field3 = value2")
        self.assertEqual(result, "SELECT field1 FROM table1 WHERE field2 = value1, field3 = value2;")

    def test_select_6(self):
        sql = SQLGenerator('table1')
        result = sql.select(['field1'])
        self.assertEqual(result, "SELECT field1 FROM table1;")



class SQLGeneratorTestInsert(unittest.TestCase):
    def test_insert(self):
        sql = SQLGenerator('table1')
        result = sql.insert({'field1': 'value1', 'field2': 'value2'})
        self.assertEqual(result, "INSERT INTO table1 (field1, field2) VALUES ('value1', 'value2');")

    def test_insert_2(self):
        sql = SQLGenerator('table1')
        result = sql.insert({'field1': 'value1', 'field2': 'value2', 'field3': 'value3'})
        self.assertEqual(result, "INSERT INTO table1 (field1, field2, field3) VALUES ('value1', 'value2', 'value3');")

    def test_insert_3(self):
        sql = SQLGenerator('table1')
        result = sql.insert({'field1': 'value1', 'field2': 'value2', 'field3': 'value3', 'field4': 'value4'})
        self.assertEqual(result,
                         "INSERT INTO table1 (field1, field2, field3, field4) VALUES ('value1', 'value2', 'value3', 'value4');")

    def test_insert_4(self):
        sql = SQLGenerator('table1')
        result = sql.insert({'field1': 'value1', 'field2': 'value2', 'field3': 'value3', 'field4': 'value4',
                             'field5': 'value5'})
        self.assertEqual(result,
                         "INSERT INTO table1 (field1, field2, field3, field4, field5) VALUES ('value1', 'value2', 'value3', 'value4', 'value5');")

    def test_insert_5(self):
        sql = SQLGenerator('table1')
        result = sql.insert({'field1': 'value1', 'field2': 'value2', 'field3': 'value3', 'field4': 'value4',
                             'field5': 'value5', 'field6': 'value6'})
        self.assertEqual(result,
                         "INSERT INTO table1 (field1, field2, field3, field4, field5, field6) VALUES ('value1', 'value2', 'value3', 'value4', 'value5', 'value6');")

class SQLGeneratorTestUpdate(unittest.TestCase):
    def test_update(self):
        sql = SQLGenerator('table1')
        result = sql.update({'field1': 'new_value1', 'field2': 'new_value2'}, "field3 = value1")
        self.assertEqual(result,
                         "UPDATE table1 SET field1 = 'new_value1', field2 = 'new_value2' WHERE field3 = value1;")

    def test_update_2(self):
        sql = SQLGenerator('table1')
        result = sql.update({'field1': 'new_value1', 'field2': 'new_value2', 'field3': 'new_value3'},
                            "field4 = value1")
        self.assertEqual(result,
                         "UPDATE table1 SET field1 = 'new_value1', field2 = 'new_value2', field3 = 'new_value3' WHERE field4 = value1;")

    def test_update_3(self):
        sql = SQLGenerator('table1')
        result = sql.update({'field1': 'new_value1', 'field2': 'new_value2', 'field3': 'new_value3',
                             'field4': 'new_value4'}, "field5 = value1")
        self.assertEqual(result,
                         "UPDATE table1 SET field1 = 'new_value1', field2 = 'new_value2', field3 = 'new_value3', field4 = 'new_value4' WHERE field5 = value1;")

    def test_update_4(self):
        sql = SQLGenerator('table1')
        result = sql.update({'field1': 'new_value1', 'field2': 'new_value2', 'field3': 'new_value3',
                             'field4': 'new_value4', 'field5': 'new_value5'}, "field6 = value1")
        self.assertEqual(result,
                         "UPDATE table1 SET field1 = 'new_value1', field2 = 'new_value2', field3 = 'new_value3', field4 = 'new_value4', field5 = 'new_value5' WHERE field6 = value1;")

    def test_update_5(self):
        sql = SQLGenerator('table1')
        result = sql.update({'field1': 'new_value1', 'field2': 'new_value2', 'field3': 'new_value3',
                             'field4': 'new_value4', 'field5': 'new_value5', 'field6': 'new_value6'},
                            "field7 = value1")
        self.assertEqual(result,
                         "UPDATE table1 SET field1 = 'new_value1', field2 = 'new_value2', field3 = 'new_value3', field4 = 'new_value4', field5 = 'new_value5', field6 = 'new_value6' WHERE field7 = value1;")

class SQLGeneratorTestDelete(unittest.TestCase):
    def test_delete(self):
        sql = SQLGenerator('table1')
        result = sql.delete("field1 = value1")
        self.assertEqual(result, "DELETE FROM table1 WHERE field1 = value1;")

    def test_delete_2(self):
        sql = SQLGenerator('table1')
        result = sql.delete("field1 = value1 AND field2 = value2")
        self.assertEqual(result, "DELETE FROM table1 WHERE field1 = value1 AND field2 = value2;")

    def test_delete_3(self):
        sql = SQLGenerator('table1')
        result = sql.delete("field1 = value1 AND field2 = value2 AND field3 = value3")
        self.assertEqual(result, "DELETE FROM table1 WHERE field1 = value1 AND field2 = value2 AND field3 = value3;")

    def test_delete_4(self):
        sql = SQLGenerator('table1')
        result = sql.delete("field1 = value1 AND field2 = value2 AND field3 = value3 AND field4 = value4")
        self.assertEqual(result,
                         "DELETE FROM table1 WHERE field1 = value1 AND field2 = value2 AND field3 = value3 AND field4 = value4;")

    def test_delete_5(self):
        sql = SQLGenerator('table1')
        result = sql.delete("field1 = value1 AND field2 = value2 AND field3 = value3 AND field4 = value4 AND field5 = value5")
        self.assertEqual(result,
                         "DELETE FROM table1 WHERE field1 = value1 AND field2 = value2 AND field3 = value3 AND field4 = value4 AND field5 = value5;")

class SQLGeneratorTestSelectFemaleUnderAge(unittest.TestCase):
    def test_select_female_under_age(self):
        sql = SQLGenerator('table1')
        result = sql.select_female_under_age(30)
        self.assertEqual(result, "SELECT * FROM table1 WHERE age < 30 AND gender = 'female';")

    def test_select_female_under_age_2(self):
        sql = SQLGenerator('table1')
        result = sql.select_female_under_age(40)
        self.assertEqual(result,"SELECT * FROM table1 WHERE age < 40 AND gender = 'female';")

    def test_select_female_under_age_3(self):
        sql = SQLGenerator('table1')
        result = sql.select_female_under_age(20)
        self.assertEqual(result,"SELECT * FROM table1 WHERE age < 20 AND gender = 'female';")

    def test_select_female_under_age_4(self):
        sql = SQLGenerator('table1')
        result = sql.select_female_under_age(10)
        self.assertEqual(result,"SELECT * FROM table1 WHERE age < 10 AND gender = 'female';")

    def test_select_female_under_age_5(self):
        sql = SQLGenerator('table1')
        result = sql.select_female_under_age(50)
        self.assertEqual(result,"SELECT * FROM table1 WHERE age < 50 AND gender = 'female';")

class SQLGeneratorTestSelectByAgeRange(unittest.TestCase):
    def test_select_by_age_range(self):
        sql = SQLGenerator('table1')
        result = sql.select_by_age_range(20, 30)
        self.assertEqual(result, "SELECT * FROM table1 WHERE age BETWEEN 20 AND 30;")

    def test_select_by_age_range_2(self):
        sql = SQLGenerator('table1')
        result = sql.select_by_age_range(10, 20)
        self.assertEqual(result, "SELECT * FROM table1 WHERE age BETWEEN 10 AND 20;")

    def test_select_by_age_range_3(self):
        sql = SQLGenerator('table1')
        result = sql.select_by_age_range(30, 40)
        self.assertEqual(result, "SELECT * FROM table1 WHERE age BETWEEN 30 AND 40;")

    def test_select_by_age_range_4(self):
        sql = SQLGenerator('table1')
        result = sql.select_by_age_range(40, 50)
        self.assertEqual(result, "SELECT * FROM table1 WHERE age BETWEEN 40 AND 50;")

    def test_select_by_age_range_5(self):
        sql = SQLGenerator('table1')
        result = sql.select_by_age_range(50, 60)
        self.assertEqual(result, "SELECT * FROM table1 WHERE age BETWEEN 50 AND 60;")


class SQLGeneratorTestMain(unittest.TestCase):
    def test_main(self):
        sql = SQLGenerator('table1')
        self.assertEqual(sql.select(['field1', 'field2'], "field3 = value1"),
                         "SELECT field1, field2 FROM table1 WHERE field3 = value1;")
        self.assertEqual(sql.insert({'field1': 'value1', 'field2': 'value2'}),
                         "INSERT INTO table1 (field1, field2) VALUES ('value1', 'value2');")
        self.assertEqual(sql.update({'field1': 'new_value1', 'field2': 'new_value2'},
                                    "field3 = value1"),
                         "UPDATE table1 SET field1 = 'new_value1', field2 = 'new_value2' WHERE field3 = value1;")
        self.assertEqual(sql.delete("field1 = value1"),
                         "DELETE FROM table1 WHERE field1 = value1;")
        self.assertEqual(sql.select_female_under_age(30),
                         "SELECT * FROM table1 WHERE age < 30 AND gender = 'female';")
        self.assertEqual(sql.select_by_age_range(20, 30),
                         "SELECT * FROM table1 WHERE age BETWEEN 20 AND 30;")

