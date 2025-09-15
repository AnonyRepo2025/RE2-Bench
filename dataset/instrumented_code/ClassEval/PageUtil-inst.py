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
       jsonl_path = json_base + "/PageUtil.jsonl"
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
# PageUtil class is a versatile utility for handling pagination and search functionalities in an efficient and convenient manner.

class PageUtil:
    def __init__(self, data, page_size):
        """
        Initialize the PageUtil object with the given data and page size.
        :param data: list, the data to be paginated
        :param page_size: int, the number of items per page
        """
        self.data = data
        self.page_size = page_size
        self.total_items = len(data)
        self.total_pages = (self.total_items + page_size - 1) // page_size

    def get_page(self, page_number):
        """
        Retrieve a specific page of data.
        :param page_number: int, the page number to fetch
        :return: list, the data on the specified page
        >>> page_util = PageUtil([1, 2, 3, 4], 1)
        >>> page_util.get_page(1)
        [1]

        """


    def get_page_info(self, page_number):
        """
        Retrieve information about a specific page.
        :param page_number: int, the page number to fetch information about
        :return: dict, containing page information such as current page number, total pages, etc.
        >>> page_util = PageUtil([1, 2, 3, 4], 1)
        >>> page_util.get_page_info(1)
        >>> {
        >>>     "current_page": 1,
        >>>     "per_page": 1,
        >>>     "total_pages": 4,
        >>>     "total_items": 4,
        >>>     "has_previous": False,
        >>>     "has_next": True,
        >>>     "data": [1]
        >>> }

        """


    def search(self, keyword):
        """
        Search for items in the data that contain the given keyword.
        :param keyword: str, the keyword to search for
        :return: dict, containing search information such as total results and matching items
        >>> page_util = PageUtil([1, 2, 3, 4], 1)
        >>> page_util.search("1")
        >>> search_info = {
        >>>     "keyword": "1",
        >>>     "total_results": 1,
        >>>     "total_pages": 1,
        >>>     "results": [1]
        >>> }
        """


'''


class PageUtil:
    def __init__(self, data, page_size):
        self.data = data
        self.page_size = page_size
        self.total_items = len(data)
        self.total_pages = (self.total_items + page_size - 1) // page_size

    @inspect_code
    def get_page(self, page_number):
        if page_number < 1 or page_number > self.total_pages:
            return []

        start_index = (page_number - 1) * self.page_size
        end_index = start_index + self.page_size
        return self.data[start_index:end_index]

    @inspect_code
    def get_page_info(self, page_number):
        if page_number < 1 or page_number > self.total_pages:
            return {}

        start_index = (page_number - 1) * self.page_size
        end_index = min(start_index + self.page_size, self.total_items)
        page_data = self.data[start_index:end_index]

        page_info = {
            "current_page": page_number,
            "per_page": self.page_size,
            "total_pages": self.total_pages,
            "total_items": self.total_items,
            "has_previous": page_number > 1,
            "has_next": page_number < self.total_pages,
            "data": page_data
        }
        return page_info

    @inspect_code
    def search(self, keyword):
        results = [item for item in self.data if keyword in str(item)]
        num_results = len(results)
        num_pages = (num_results + self.page_size - 1) // self.page_size

        search_info = {
            "keyword": keyword,
            "total_results": num_results,
            "total_pages": num_pages,
            "results": results
        }
        return search_info



import unittest


class PageUtilTestGetPage(unittest.TestCase):
    def setUp(self):
        self.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.page_size = 3
        self.page_util = PageUtil(self.data, self.page_size)

    def test_get_page_1(self):
        page_number = 1
        expected_page = [1, 2, 3]
        actual_page = self.page_util.get_page(page_number)
        self.assertEqual(actual_page, expected_page)

    def test_get_page_2(self):
        page_number = 2
        expected_page = [4, 5, 6]
        actual_page = self.page_util.get_page(page_number)
        self.assertEqual(actual_page, expected_page)

    def test_get_page_3(self):
        page_number = 3
        expected_page = [7, 8, 9]
        actual_page = self.page_util.get_page(page_number)
        self.assertEqual(actual_page, expected_page)

    def test_get_page_4(self):
        page_number = 4
        expected_page = [10]
        actual_page = self.page_util.get_page(page_number)
        self.assertEqual(actual_page, expected_page)

    def test_get_page_5(self):
        invalid_page_number = 0
        empty_page = []
        actual_page = self.page_util.get_page(invalid_page_number)
        self.assertEqual(actual_page, empty_page)


class PageUtilTestGetPageInfo(unittest.TestCase):
    def setUp(self):
        self.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.page_size = 3
        self.page_util = PageUtil(self.data, self.page_size)

    def test_get_page_info_1(self):
        page_number = 2
        expected_info = {
            "current_page": 2,
            "per_page": 3,
            "total_pages": 4,
            "total_items": 10,
            "has_previous": True,
            "has_next": True,
            "data": [4, 5, 6]
        }
        actual_info = self.page_util.get_page_info(page_number)
        self.assertEqual(actual_info, expected_info)

    def test_get_page_info_2(self):
        page_number = 1
        expected_info = {
            "current_page": 1,
            "per_page": 3,
            "total_pages": 4,
            "total_items": 10,
            "has_previous": False,
            "has_next": True,
            "data": [1, 2, 3]
        }
        actual_info = self.page_util.get_page_info(page_number)
        self.assertEqual(actual_info, expected_info)

    def test_get_page_info_3(self):
        page_number = 3
        expected_info = {
            "current_page": 3,
            "per_page": 3,
            "total_pages": 4,
            "total_items": 10,
            "has_previous": True,
            "has_next": True,
            "data": [7, 8, 9]
        }
        actual_info = self.page_util.get_page_info(page_number)
        self.assertEqual(actual_info, expected_info)

    def test_get_page_info_4(self):
        page_number = 4
        expected_info = {
            "current_page": 4,
            "per_page": 3,
            "total_pages": 4,
            "total_items": 10,
            "has_previous": True,
            "has_next": False,
            "data": [10]
        }
        actual_info = self.page_util.get_page_info(page_number)
        self.assertEqual(actual_info, expected_info)

    def test_get_page_info_5(self):
        invalid_page_number = 5
        empty_info = {}
        actual_info = self.page_util.get_page_info(invalid_page_number)
        self.assertEqual(actual_info, empty_info)


class PageUtilTestSearch(unittest.TestCase):
    def setUp(self):
        self.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.page_size = 3
        self.page_util = PageUtil(self.data, self.page_size)

    def test_search_1(self):
        keyword = "1"
        expected_results = {
            "keyword": "1",
            "total_results": 2,
            "total_pages": 1,
            "results": [1, 10]
        }
        actual_results = self.page_util.search(keyword)
        self.assertEqual(actual_results, expected_results)

    def test_search_2(self):
        keyword = "2"
        expected_results = {
            "keyword": "2",
            "total_results": 1,
            "total_pages": 1,
            "results": [2]
        }
        actual_results = self.page_util.search(keyword)
        self.assertEqual(actual_results, expected_results)

    def test_search_3(self):
        keyword = "3"
        expected_results = {
            "keyword": "3",
            "total_results": 1,
            "total_pages": 1,
            "results": [3]
        }
        actual_results = self.page_util.search(keyword)
        self.assertEqual(actual_results, expected_results)

    def test_search_4(self):
        keyword = "4"
        expected_results = {
            "keyword": "4",
            "total_results": 1,
            "total_pages": 1,
            "results": [4]
        }
        actual_results = self.page_util.search(keyword)
        self.assertEqual(actual_results, expected_results)

    def test_search_5(self):
        keyword = "11"
        expected_results = {
            "keyword": "11",
            "total_results": 0,
            "total_pages": 0,
            "results": []
        }
        actual_results = self.page_util.search(keyword)
        self.assertEqual(actual_results, expected_results)


class PageUtilTest(unittest.TestCase):
    def setUp(self):
        self.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.page_size = 3
        self.page_util = PageUtil(self.data, self.page_size)

    def test_pageutil(self):
        page_number = 1
        expected_page = [1, 2, 3]
        actual_page = self.page_util.get_page(page_number)
        self.assertEqual(actual_page, expected_page)

        page_number = 2
        expected_info = {
            "current_page": 2,
            "per_page": 3,
            "total_pages": 4,
            "total_items": 10,
            "has_previous": True,
            "has_next": True,
            "data": [4, 5, 6]
        }
        actual_info = self.page_util.get_page_info(page_number)
        self.assertEqual(actual_info, expected_info)

        keyword = "4"
        expected_results = {
            "keyword": "4",
            "total_results": 1,
            "total_pages": 1,
            "results": [4]
        }
        actual_results = self.page_util.search(keyword)
        self.assertEqual(actual_results, expected_results)
