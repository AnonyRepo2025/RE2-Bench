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
       jsonl_path = json_base + "/TimeUtils.jsonl"
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
# This is a time util class, including getting the current time and date, adding seconds to a datetime, converting between strings and datetime objects, calculating the time difference in minutes, and formatting a datetime object.

import datetime
import time

class TimeUtils:

    def __init__(self):
        """
        Get the current datetime
        """
        self.datetime = datetime.datetime.now()

    def get_current_time(self):
        """
        Return the current time in the format of '%H:%M:%S'
        :return: string
        >>> timeutils = TimeUtils()
        >>> timeutils.get_current_time()
        "19:19:22"
        """

    def get_current_date(self):
        """
        Return the current date in the format of "%Y-%m-%d"
        :return: string
        >>> timeutils.get_current_date()
        "2023-06-14"
        """

    def add_seconds(self, seconds):
        """
        Add the specified number of seconds to the current time
        :param seconds: int, number of seconds to add
        :return: string, time after adding the specified number of seconds in the format '%H:%M:%S'
        >>> timeutils.add_seconds(600)
        "19:29:22"
        """

    def string_to_datetime(self, string):
        """
        Convert the time string to a datetime instance
        :param string: string, string before converting format
        :return: datetime instance
        >>> timeutils.string_to_datetime("2001-7-18 1:1:1")
        2001-07-18 01:01:01
        """

    def datetime_to_string(self, datetime):
        """
        Convert a datetime instance to a string
        :param datetime: the datetime instance to convert
        :return: string, converted time string
        >>> timeutils.datetime_to_string(timeutils.datetime)
        "2023-06-14 19:30:03"
        """

    def get_minutes(self, string_time1, string_time2):
        """
        Calculate how many minutes have passed between two times, and round the results to the nearest
        :return: int, the number of minutes between two times, rounded off
        >>> timeutils.get_minutes("2001-7-18 1:1:1", "2001-7-18 2:1:1")
        60
        """

    def get_format_time(self, year, month, day, hour, minute, second):
        """
        get format time
        :param year: int
        :param month: int
        :param day: int
        :param hour: int
        :param minute: int
        :param second: int
        :return: formatted time string
        >>> timeutils.get_format_time(2001, 7, 18, 1, 1, 1)
        "2001-07-18 01:01:01"
        """
'''

import datetime
import time

class TimeUtils:

    def __init__(self):
        self.datetime = datetime.datetime.now()

    @inspect_code
    def get_current_time(self):
        format = "%H:%M:%S"
        return self.datetime.strftime(format)

    @inspect_code
    def get_current_date(self):
        format = "%Y-%m-%d"
        return self.datetime.strftime(format)

    @inspect_code
    def add_seconds(self, seconds):
        new_datetime = self.datetime + datetime.timedelta(seconds=seconds)
        format = "%H:%M:%S"
        return new_datetime.strftime(format)

    @inspect_code
    def string_to_datetime(self, string):
        return datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S")

    @inspect_code
    def datetime_to_string(self, datetime):
        return datetime.strftime("%Y-%m-%d %H:%M:%S")

    @inspect_code
    def get_minutes(self, string_time1, string_time2):
        time1 = self.string_to_datetime(string_time1)
        time2 = self.string_to_datetime(string_time2)
        return round((time2 - time1).seconds / 60)

    @inspect_code
    def get_format_time(self, year, month, day, hour, minute, second):
        format = "%Y-%m-%d %H:%M:%S"
        time_item = datetime.datetime(year, month, day, hour, minute, second)
        return time_item.strftime(format)





import unittest


class TimeUtilsTestGetCurrentTime(unittest.TestCase):
    def test_get_current_time_1(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_current_time(), timeutils.datetime.strftime("%H:%M:%S"))

    def test_get_current_time_2(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_current_time(), timeutils.datetime.strftime("%H:%M:%S"))

    def test_get_current_time_3(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_current_time(), timeutils.datetime.strftime("%H:%M:%S"))

    def test_get_current_time_4(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_current_time(), timeutils.datetime.strftime("%H:%M:%S"))

    def test_get_current_time_5(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_current_time(), timeutils.datetime.strftime("%H:%M:%S"))


class TimeUtilsTestGetCurrentDate(unittest.TestCase):
    def test_get_current_date_1(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_current_date(), timeutils.datetime.strftime("%Y-%m-%d"))

    def test_get_current_date_2(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_current_date(), timeutils.datetime.strftime("%Y-%m-%d"))

    def test_get_current_date_3(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_current_date(), timeutils.datetime.strftime("%Y-%m-%d"))

    def test_get_current_date_4(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_current_date(), timeutils.datetime.strftime("%Y-%m-%d"))

    def test_get_current_date_5(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_current_date(), timeutils.datetime.strftime("%Y-%m-%d"))


class TimeUtilsTestAddSeconds(unittest.TestCase):
    def test_add_seconds_1(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.add_seconds(600),
                         (timeutils.datetime + datetime.timedelta(seconds=600)).strftime("%H:%M:%S"))

    def test_add_seconds_2(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.add_seconds(500),
                         (timeutils.datetime + datetime.timedelta(seconds=500)).strftime("%H:%M:%S"))

    def test_add_seconds_3(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.add_seconds(400),
                         (timeutils.datetime + datetime.timedelta(seconds=400)).strftime("%H:%M:%S"))

    def test_add_seconds_4(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.add_seconds(300),
                         (timeutils.datetime + datetime.timedelta(seconds=300)).strftime("%H:%M:%S"))

    def test_add_seconds_5(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.add_seconds(200),
                         (timeutils.datetime + datetime.timedelta(seconds=200)).strftime("%H:%M:%S"))


class TimeUtilsTestStringToDatetime(unittest.TestCase):
    def test_string_to_datetime_1(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.string_to_datetime('2001-7-18 1:1:1'), datetime.datetime(2001, 7, 18, 1, 1, 1))

    def test_string_to_datetime_2(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.string_to_datetime('2001-7-17 1:1:1'), datetime.datetime(2001, 7, 17, 1, 1, 1))

    def test_string_to_datetime_3(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.string_to_datetime('2001-7-16 1:1:1'), datetime.datetime(2001, 7, 16, 1, 1, 1))

    def test_string_to_datetime_4(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.string_to_datetime('2001-7-15 1:1:1'), datetime.datetime(2001, 7, 15, 1, 1, 1))

    def test_string_to_datetime_5(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.string_to_datetime('2001-7-14 1:1:1'), datetime.datetime(2001, 7, 14, 1, 1, 1))


class TimeUtilsTestDatetimeToString(unittest.TestCase):
    def test_datetime_to_string_1(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.datetime_to_string(timeutils.datetime),
                         timeutils.datetime.strftime("%Y-%m-%d %H:%M:%S"))

    def test_datetime_to_string_2(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.datetime_to_string(timeutils.datetime),
                         timeutils.datetime.strftime("%Y-%m-%d %H:%M:%S"))

    def test_datetime_to_string_3(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.datetime_to_string(timeutils.datetime),
                         timeutils.datetime.strftime("%Y-%m-%d %H:%M:%S"))

    def test_datetime_to_string_4(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.datetime_to_string(timeutils.datetime),
                         timeutils.datetime.strftime("%Y-%m-%d %H:%M:%S"))

    def test_datetime_to_string_5(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.datetime_to_string(timeutils.datetime),
                         timeutils.datetime.strftime("%Y-%m-%d %H:%M:%S"))


class TimeUtilsTestGetMinutes(unittest.TestCase):
    def test_get_minutes_1(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_minutes("2001-7-18 1:1:1", "2001-7-18 2:1:1"), 60)

    def test_get_minutes_2(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_minutes("2001-7-18 1:1:1", "2001-7-18 3:1:1"), 120)

    def test_get_minutes_3(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_minutes("2001-7-18 1:1:1", "2001-7-18 4:1:1"), 180)

    def test_get_minutes_4(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_minutes("2001-7-18 1:1:1", "2001-7-18 5:1:1"), 240)

    def test_get_minutes_5(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_minutes("2001-7-18 1:1:1", "2001-7-18 6:1:1"), 300)


class TimeUtilsTestGetFormatTime(unittest.TestCase):
    def test_get_format_time_1(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_format_time(2001, 7, 18, 1, 1, 1), "2001-07-18 01:01:01")

    def test_get_format_time_2(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_format_time(2001, 7, 17, 1, 1, 1), "2001-07-17 01:01:01")

    def test_get_format_time_3(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_format_time(2001, 7, 16, 1, 1, 1), "2001-07-16 01:01:01")

    def test_get_format_time_4(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_format_time(2001, 7, 15, 1, 1, 1), "2001-07-15 01:01:01")

    def test_get_format_time_5(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_format_time(2001, 7, 14, 1, 1, 1), "2001-07-14 01:01:01")


class TimeUtilsTest(unittest.TestCase):
    def test_timeutils(self):
        timeutils = TimeUtils()
        self.assertEqual(timeutils.get_current_time(), timeutils.datetime.strftime("%H:%M:%S"))
        self.assertEqual(timeutils.get_current_date(), timeutils.datetime.strftime("%Y-%m-%d"))
        self.assertEqual(timeutils.add_seconds(600),
                         (timeutils.datetime + datetime.timedelta(seconds=600)).strftime("%H:%M:%S"))
        self.assertEqual(timeutils.string_to_datetime('2001-7-18 1:1:1'), datetime.datetime(2001, 7, 18, 1, 1, 1))
        self.assertEqual(timeutils.datetime_to_string(timeutils.datetime),
                         timeutils.datetime.strftime("%Y-%m-%d %H:%M:%S"))
        self.assertEqual(timeutils.get_minutes("2001-7-18 1:1:1", "2001-7-18 2:1:1"), 60)
        self.assertEqual(timeutils.get_format_time(2001, 7, 18, 1, 1, 1), "2001-07-18 01:01:01")
