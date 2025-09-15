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
       jsonl_path = json_base + "/Thermostat.jsonl"
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
# The class manages temperature control, including setting and retrieving the target temperature, adjusting the mode, and simulating temperature operation.

import time

class Thermostat:
    def __init__(self, current_temperature, target_temperature, mode):
        """
        initialize instances of the Thermostat class, including the current temperature, target temperature, and operating mode.
        :param current_temperature: float
        :param target_temperature: float
        :param mode: str, the work mode
        """
        self.current_temperature = current_temperature
        self.target_temperature = target_temperature
        self.mode = mode

    def get_target_temperature(self):
        """
        Get the target temperature of an instance of the Thermostat class.
        :return self.current_temperature: int
        >>> thermostat.get_target_temperature()
        37.5
        """

    def set_target_temperature(self, temperature):
        """
        Set the target temperature
        :param temperature: float, the target temperature
        >>> thermostat.set_target_temperature(37.6)
        >>> thermostat.target_temperature
        37.6
        """

    def get_mode(self):
        """
        Get the current work mode
        :return mode: str, working mode. only ['heat', 'cool']
        """

    def set_mode(self, mode):
        """
        Get the current work mode
        :param mode: str, working mode. only ['heat', 'cool']
        >>> thermostat.set_mode('cool')
        >>> thermostat.mode
        'cool'
        """

    def auto_set_mode(self):
        """
        Automatically set the operating mode by comparing with the current temperature and target temperature. If the current temperature is lower than the target temperature, the operating mode is set to 'heat', otherwise it is set to 'cool'.
        >>> thermostat = Thermostat(20.4, 37.5, 'cool')
        >>> thermostat.auto_set_mode()
        >>> thermostat.mode
        'heat'
        """

    def auto_check_conflict(self):
        """
        Check if there is a conflict between the operating mode and the relationship between the current temperature and the target temperature.
        If there is a conflict, the operating mode will be adjusted automatically.
        :return: True if mode isn't conflict with the relationship between the current temperature and the target temperature, or False otherwise.
        >>> thermostat = Thermostat(20.4, 37.5, 'cool')
        >>> thermostat.auto_check_conflict()
        False
        >>> thermostat.mode
        'heat'
        """

    def simulate_operation(self):
        """
        simulate the operation of Thermostat. It will automatically start the auto_set_mode method to set the operating mode,
        and then automatically adjust the current temperature according to the operating mode until the target temperature is reached.
        :return time: int, the time it took to complete the simulation.
        >>> thermostat = Thermostat(20.4, 37.5, 'cool')
        >>> thermostat.simulate_operation()
        18
        """


'''

import time

class Thermostat:
    def __init__(self, current_temperature, target_temperature, mode):
        self.current_temperature = current_temperature
        self.target_temperature = target_temperature
        self.mode = mode

    @inspect_code
    def get_target_temperature(self):
        return self.target_temperature

    @inspect_code
    def set_target_temperature(self, temperature):
        self.target_temperature = temperature

    @inspect_code
    def get_mode(self):
        return self.mode

    @inspect_code
    def set_mode(self, mode):
        if mode in ['heat', 'cool']:
            self.mode = mode
        else:
            return False

    @inspect_code
    def auto_set_mode(self):
        if self.current_temperature < self.target_temperature:
            self.mode = 'heat'
        else:
            self.mode = 'cool'

    @inspect_code
    def auto_check_conflict(self):
        if self.current_temperature > self.target_temperature:
            if self.mode == 'cool':
                return True
            else:
                self.auto_set_mode()
                return False
        else:
            if self.mode == 'heat':
                return True
            else:
                self.auto_set_mode()
                return False

    @inspect_code
    def simulate_operation(self):
        self.auto_set_mode()
        use_time = 0
        if self.mode == 'heat':
            while(self.current_temperature < self.target_temperature):
                self.current_temperature += 1
                use_time += 1
        else:
            while(self.current_temperature > self.target_temperature):
                self.current_temperature -= 1
                use_time += 1
        return use_time

import unittest

class ThermostatTestGetTargetTemperature(unittest.TestCase):
    def test_get_target_temperature_1(self):
        t = Thermostat(20, 25, 'heat')
        self.assertEqual(t.get_target_temperature(), 25)

    def test_get_target_temperature_2(self):
        t = Thermostat(20, 25, 'cool')
        self.assertEqual(t.get_target_temperature(), 25)

    def test_get_target_temperature_3(self):
        t = Thermostat(20, 25, 'test')
        self.assertEqual(t.get_target_temperature(), 25)

    def test_get_target_temperature_4(self):
        t = Thermostat(25, 25, 'cool')
        self.assertEqual(t.get_target_temperature(), 25)

    def test_get_target_temperature_5(self):
        t = Thermostat(25, 25, 'heat')
        self.assertEqual(t.get_target_temperature(), 25)


class ThermostatTestSetTargetTemperature(unittest.TestCase):
    def test_set_target_temperature_1(self):
        t = Thermostat(20, 25, 'heat')
        t.set_target_temperature(30)
        self.assertEqual(t.get_target_temperature(), 30)

    def test_set_target_temperature_2(self):
        t = Thermostat(20, 25, 'cool')
        t.set_target_temperature(10)
        self.assertEqual(t.get_target_temperature(), 10)

    def test_set_target_temperature_3(self):
        t = Thermostat(20, 25, 'test')
        t.set_target_temperature(10)
        self.assertEqual(t.get_target_temperature(), 10)

    def test_set_target_temperature_4(self):
        t = Thermostat(25, 25, 'cool')
        t.set_target_temperature(10)
        self.assertEqual(t.get_target_temperature(), 10)

    def test_set_target_temperature_5(self):
        t = Thermostat(25, 25, 'heat')
        t.set_target_temperature(10)
        self.assertEqual(t.get_target_temperature(), 10)


class ThermostatTestGetMode(unittest.TestCase):
    def test_get_mode_1(self):
        t = Thermostat(20, 25, 'heat')
        self.assertEqual(t.get_mode(), 'heat')

    def test_get_mode_2(self):
        t = Thermostat(20, 25, 'cool')
        self.assertEqual(t.get_mode(), 'cool')

    def test_get_mode_3(self):
        t = Thermostat(20, 25, 'test')
        self.assertEqual(t.get_mode(), 'test')

    def test_get_mode_4(self):
        t = Thermostat(25, 25, 'cool')
        self.assertEqual(t.get_mode(), 'cool')

    def test_get_mode_5(self):
        t = Thermostat(25, 25, 'heat')
        self.assertEqual(t.get_mode(), 'heat')


class ThermostatTestSetMode(unittest.TestCase):
    def test_set_mode_1(self):
        t = Thermostat(20, 25, 'heat')
        t.set_mode('cool')
        self.assertEqual(t.get_mode(), 'cool')

    # use mode that not in ['heat', 'cool']
    def test_set_mode_2(self):
        t = Thermostat(20, 25, 'heat')
        self.assertFalse(t.set_mode('test'))

    def test_set_mode_3(self):
        t = Thermostat(20, 25, 'cool')
        t.set_mode('heat')
        self.assertEqual(t.get_mode(), 'heat')

    def test_set_mode_4(self):
        t = Thermostat(20, 25, 'test')
        t.set_mode('heat')
        self.assertEqual(t.get_mode(), 'heat')

    def test_set_mode_5(self):
        t = Thermostat(25, 25, 'cool')
        t.set_mode('heat')
        self.assertEqual(t.get_mode(), 'heat')


class ThermostatTestAutoSetMode(unittest.TestCase):
    def test_auto_set_mode_1(self):
        t = Thermostat(20, 25, 'heat')
        t.auto_set_mode()
        self.assertEqual(t.get_mode(), 'heat')

    def test_auto_set_mode_2(self):
        t = Thermostat(25, 20, 'heat')
        t.auto_set_mode()
        self.assertEqual(t.get_mode(), 'cool')

    def test_auto_set_mode_3(self):
        t = Thermostat(25, 20, 'cool')
        t.auto_set_mode()
        self.assertEqual(t.get_mode(), 'cool')

    def test_auto_set_mode_4(self):
        t = Thermostat(20, 25, 'cool')
        t.auto_set_mode()
        self.assertEqual(t.get_mode(), 'heat')

    def test_auto_set_mode_5(self):
        t = Thermostat(25, 25, 'cool')
        t.auto_set_mode()
        self.assertEqual(t.get_mode(), 'cool')

class ThermostatTestAutoCheckConflict(unittest.TestCase):
    def test_auto_check_conflict_1(self):
        t = Thermostat(30, 25, 'cool')
        self.assertTrue(t.auto_check_conflict())

    def test_auto_check_conflict_2(self):
        t = Thermostat(30, 25, 'heat')
        self.assertFalse(t.auto_check_conflict())
        self.assertEqual(t.mode, 'cool')

    def test_auto_check_conflict_3(self):
        t = Thermostat(25, 30, 'heat')
        self.assertTrue(t.auto_check_conflict())

    def test_auto_check_conflict_4(self):
        t = Thermostat(25, 30, 'cool')
        self.assertFalse(t.auto_check_conflict())
        self.assertEqual(t.mode, 'heat')

    def test_auto_check_conflict_5(self):
        t = Thermostat(25, 25, 'cool')
        self.assertFalse(t.auto_check_conflict())
        self.assertEqual(t.mode, 'cool')


class ThermostatTestSimulateOperation(unittest.TestCase):
    def test_simulate_operation_1(self):
        t = Thermostat(20, 25, 'heat')
        self.assertEqual(t.simulate_operation(), 5)
        self.assertEqual(t.get_mode(), 'heat')
        self.assertEqual(t.current_temperature, 25)

    def test_simulate_operation_2(self):
        t = Thermostat(25.7, 20, 'cool')
        self.assertEqual(t.simulate_operation(), 6)
        self.assertEqual(t.get_mode(), 'cool')
        self.assertEqual(t.current_temperature, 19.7)

    def test_simulate_operation_3(self):
        t = Thermostat(25, 25, 'heat')
        self.assertEqual(t.simulate_operation(), 0)
        self.assertEqual(t.get_mode(), 'cool')
        self.assertEqual(t.current_temperature, 25)

    def test_simulate_operation_4(self):
        t = Thermostat(25, 25, 'cool')
        self.assertEqual(t.simulate_operation(), 0)
        self.assertEqual(t.get_mode(), 'cool')
        self.assertEqual(t.current_temperature, 25)

    def test_simulate_operation_5(self):
        t = Thermostat(25, 25, 'test')
        self.assertEqual(t.simulate_operation(), 0)
        self.assertEqual(t.get_mode(), 'cool')
        self.assertEqual(t.current_temperature, 25)

class ThermostatTestMain(unittest.TestCase):
    def test_main(self):
        t = Thermostat(20, 37.5, 'cool')
        self.assertEqual(t.get_target_temperature(), 37.5)

        t.set_target_temperature(37.6)
        self.assertEqual(t.target_temperature, 37.6)

        self.assertEqual(t.get_mode(), 'cool')
        self.assertFalse(t.set_mode('test'))

        self.assertFalse(t.auto_check_conflict())
        self.assertEqual(t.get_mode(), 'heat')
        self.assertEqual(t.simulate_operation(), 18)

