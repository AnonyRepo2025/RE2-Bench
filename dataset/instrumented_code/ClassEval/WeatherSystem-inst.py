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
       jsonl_path = json_base + "/WeatherSystem.jsonl"
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
# This is a class representing a weather system that provides functionality to query weather information for a specific city and convert temperature units between Celsius and Fahrenheit.

class WeatherSystem:
    def __init__(self, city) -> None:
        """
        Initialize the weather system with a city name.
        """
        self.temperature = None
        self.weather = None
        self.city = city
        self.weather_list = {}

    def query(self, weather_list, tmp_units = 'celsius'):
        """
        Query the weather system for the weather and temperature of the city,and convert the temperature units based on the input parameter.
        :param weather_list: a dictionary of weather information for different cities,dict.
        :param tmp_units: the temperature units to convert to, str.
        :return: the temperature and weather of the city, tuple.
        >>> weatherSystem = WeatherSystem('New York')
        >>> weather_list = {'New York': {'weather': 'sunny','temperature': 27,'temperature units': 'celsius'},'Beijing': {'weather': 'cloudy','temperature': 23,'temperature units': 'celsius'}}
        >>> weatherSystem.query(weather_list)
        (27, 'sunny')

        """

    def set_city(self, city):
        """
        Set the city of the weather system.
        :param city: the city to set, str.
        :return: None
        >>> weatherSystem = WeatherSystem('New York')
        >>> weatherSystem.set_city('Beijing')
        >>> weatherSystem.city
        'Beijing'

        """

    def celsius_to_fahrenheit(self):
        """
        Convert the temperature from Celsius to Fahrenheit.
        :return: the temperature in Fahrenheit, float.
        >>> weatherSystem = WeatherSystem('New York')
        >>> weatherSystem.temperature = 27
        >>> weatherSystem.celsius_to_fahrenheit()
        80.6

        """

    def fahrenheit_to_celsius(self):
        """
        Convert the temperature from Fahrenheit to Celsius.
        :return: the temperature in Celsius, float.
        >>> weatherSystem = WeatherSystem('New York')
        >>> weatherSystem.temperature = 80.6
        >>> weatherSystem.fahrenheit_to_celsius()
        26.999999999999996

        """
'''

class WeatherSystem:
    def __init__(self, city) -> None:
        self.temperature = None
        self.weather = None
        self.city = city
        self.weather_list = {}
    
    @inspect_code
    def query(self, weather_list, tmp_units = 'celsius'):
        self.weather_list = weather_list
        if self.city not in weather_list:
            return False
        else:
            self.temperature = self.weather_list[self.city]['temperature']
            self.weather = self.weather_list[self.city]['weather']
        if self.weather_list[self.city]['temperature units'] != tmp_units:
            if tmp_units == 'celsius':
                return self.fahrenheit_to_celsius(), self.weather
            elif tmp_units == 'fahrenheit':
                return self.celsius_to_fahrenheit(), self.weather
        else:
            return self.temperature, self.weather
    
    @inspect_code
    def set_city(self, city):
        self.city = city

    @inspect_code
    def celsius_to_fahrenheit(self):
        return (self.temperature * 9/5) + 32

    @inspect_code
    def fahrenheit_to_celsius(self):
        return (self.temperature - 32) * 5/9

import unittest


class WeatherSystemTestQuery(unittest.TestCase):
    def test_query(self):
        weatherSystem = WeatherSystem('New York')
        weather_list = {
            'New York': {
                'weather': 'sunny',
                'temperature': 27,
                'temperature units': 'celsius'
            },
            'Beijing': {
                'weather': 'cloudy',
                'temperature': 23,
                'temperature units': 'celsius'
            }
        }
        self.assertEqual(weatherSystem.query(weather_list), (27, 'sunny'))

    def test_query_2(self):
        weatherSystem = WeatherSystem('Shanghai')
        weather_list = {
            'New York': {
                'weather': 'sunny',
                'temperature': 27,
                'temperature units': 'celsius'
            },
            'Beijing': {
                'weather': 'cloudy',
                'temperature': 23,
                'temperature units': 'celsius'
            }
        }
        self.assertEqual(weatherSystem.query(weather_list), False)

    def test_query_3(self):
        weatherSystem = WeatherSystem('Beijing')
        weather_list = {
            'New York': {
                'weather': 'sunny',
                'temperature': 27,
                'temperature units': 'celsius'
            },
            'Beijing': {
                'weather': 'cloudy',
                'temperature': 23,
                'temperature units': 'celsius'
            }
        }
        self.assertEqual(weatherSystem.query(weather_list, 'fahrenheit'), (73.4, 'cloudy'))

    def test_query_4(self):
        weatherSystem = WeatherSystem('Beijing')
        weather_list = {
            'New York': {
                'weather': 'sunny',
                'temperature': 73.47,
                'temperature units': 'fahrenheit'
            },
            'Beijing': {
                'weather': 'cloudy',
                'temperature': 73.4,
                'temperature units': 'fahrenheit'
            }
        }
        self.assertEqual(weatherSystem.query(weather_list, 'celsius'), (23.000000000000004, 'cloudy'))

    def test_query_5(self):
        weatherSystem = WeatherSystem('New York')
        weather_list = {
            'New York': {
                'weather': 'sunny',
                'temperature': 80.6,
                'temperature units': 'fahrenheit'
            },
            'Beijing': {
                'weather': 'cloudy',
                'temperature': 23,
                'temperature units': 'celsius'
            }
        }
        self.assertEqual(weatherSystem.query(weather_list, tmp_units='celsius'), (26.999999999999996, 'sunny'))

    def test_query_6(self):
        weatherSystem = WeatherSystem('New York')
        weather_list = {
            'New York': {
                'weather': 'sunny',
                'temperature': 27,
                'temperature units': 'celsius'
            },
            'Beijing': {
                'weather': 'cloudy',
                'temperature': 23,
                'temperature units': 'celsius'
            }
        }
        self.assertEqual(weatherSystem.query(weather_list, tmp_units='fahrenheit'), (80.6, 'sunny'))


class WeatherSystemTestSetCity(unittest.TestCase):
    def test_set_city(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.set_city('Beijing')
        self.assertEqual(weatherSystem.city, 'Beijing')

    def test_set_city_2(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.set_city('Shanghai')
        self.assertEqual(weatherSystem.city, 'Shanghai')

    def test_set_city_3(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.set_city('Shanghai')
        self.assertNotEqual(weatherSystem.city, 'Beijing')

    def test_set_city_4(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.set_city('Shanghai')
        self.assertNotEqual(weatherSystem.city, 'New York')

    def test_set_city_5(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.set_city('Shanghai')
        self.assertNotEqual(weatherSystem.city, 'Tokyo')


class WeatherSystemTestCelsiusToFahrenheit(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.temperature = 27
        self.assertEqual(weatherSystem.celsius_to_fahrenheit(), 80.6)

    def test_celsius_to_fahrenheit_2(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.temperature = 23
        self.assertEqual(weatherSystem.celsius_to_fahrenheit(), 73.4)

    def test_celsius_to_fahrenheit_3(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.temperature = 23
        self.assertNotEqual(weatherSystem.celsius_to_fahrenheit(), 80.6)

    def test_celsius_to_fahrenheit_4(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.temperature = 27
        self.assertNotEqual(weatherSystem.celsius_to_fahrenheit(), 73.4)

    def test_celsius_to_fahrenheit_5(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.temperature = 27
        self.assertNotEqual(weatherSystem.celsius_to_fahrenheit(), 23)


class WeatherSystemTestFahrenheitToCelsius(unittest.TestCase):
    def test_fahrenheit_to_celsius(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.temperature = 80.6
        self.assertEqual(weatherSystem.fahrenheit_to_celsius(), 26.999999999999996)

    def test_fahrenheit_to_celsius_2(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.temperature = 73.4
        self.assertEqual(weatherSystem.fahrenheit_to_celsius(), 23.000000000000004)

    def test_fahrenheit_to_celsius_3(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.temperature = 80
        self.assertNotEqual(weatherSystem.fahrenheit_to_celsius(), 23)

    def test_fahrenheit_to_celsius_4(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.temperature = 73
        self.assertNotEqual(weatherSystem.fahrenheit_to_celsius(), 27)

    def test_fahrenheit_to_celsius_5(self):
        weatherSystem = WeatherSystem('New York')
        weatherSystem.temperature = 80
        self.assertNotEqual(weatherSystem.fahrenheit_to_celsius(), 27)


class WeatherSystemTestMain(unittest.TestCase):
    def test_main(self):
        weatherSystem = WeatherSystem('New York')
        weather_list = {
            'New York': {
                'weather': 'sunny',
                'temperature': 27,
                'temperature units': 'celsius'
            },
            'Beijing': {
                'weather': 'cloudy',
                'temperature': 23,
                'temperature units': 'celsius'
            }
        }
        self.assertEqual(weatherSystem.query(weather_list), (27, 'sunny'))
        weatherSystem.set_city('Beijing')
        self.assertEqual(weatherSystem.city, 'Beijing')
        weatherSystem.temperature = 27
        self.assertEqual(weatherSystem.celsius_to_fahrenheit(), 80.6)
        weatherSystem.temperature = 80.6
        self.assertEqual(weatherSystem.fahrenheit_to_celsius(), 26.999999999999996)
