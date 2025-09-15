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
       jsonl_path = json_base + "/AutomaticGuitarSimulator.jsonl"
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
# This class is an automatic guitar simulator that can interpret and play based on the input guitar sheet music.

class AutomaticGuitarSimulator:
    def __init__(self, text) -> None:
        """
        Initialize the score to be played
        :param text:str, score to be played
        """
        self.play_text = text

    def interpret(self, display=False):
        """
        Interpret the music score to be played
        :param display:Bool, representing whether to print the interpreted score
        :return: list of dict, The dict includes two fields, Chord and Tune, which are letters and numbers, respectively. If the input is empty or contains only whitespace, an empty list is returned.
        >>> context = AutomaticGuitarSimulator("C53231323 Em43231323 F43231323 G63231323")
        >>> play_list = context.interpret(display = False)
        [{'Chord': 'C', 'Tune': '53231323'}, {'Chord': 'Em', 'Tune': '43231323'}, {'Chord': 'F', 'Tune': '43231323'}, {'Chord': 'G', 'Tune': '63231323'}]

        """


    def display(self, key, value):
        """
        Print out chord and play tune with following format: Normal Guitar Playing -- Chord: %s, Play Tune: %s
        :param key:str, chord
        :param value:str, play tune
        :return: str
        >>> context = AutomaticGuitarSimulator("C53231323 Em43231323 F43231323 G63231323")
        >>> context.display("C", "53231323")
        Normal Guitar Playing -- Chord: C, Play Tune: 53231323

        """


'''


class AutomaticGuitarSimulator:
    def __init__(self, text) -> None:
        self.play_text = text

    @inspect_code
    def interpret(self, display=False):
        if not self.play_text.strip():
            return []
        else:
            play_list = []
            play_segs = self.play_text.split(" ")
            for play_seg in play_segs:
                pos = 0
                for ele in play_seg:
                    if ele.isalpha():
                        pos += 1
                        continue
                    break
                play_chord = play_seg[0:pos]
                play_value = play_seg[pos:]
                play_list.append({'Chord': play_chord, 'Tune': play_value})
                if display:
                    self.display(play_chord, play_value)
            return play_list

    @inspect_code
    def display(self, key, value):
        return "Normal Guitar Playing -- Chord: %s, Play Tune: %s" % (key, value)



import unittest


class AutomaticGuitarSimulatorTestInterpret(unittest.TestCase):
    def test_interpret_1(self):
        context = AutomaticGuitarSimulator("C53231323")
        play_list = context.interpret()
        self.assertEqual(play_list, [{'Chord': 'C', 'Tune': '53231323'}])

    def test_interpret_2(self):
        context = AutomaticGuitarSimulator("F43231323")
        play_list = context.interpret()
        self.assertEqual(play_list, [{'Chord': 'F', 'Tune': '43231323'}])

    def test_interpret_3(self):
        context = AutomaticGuitarSimulator("Em43231323")
        play_list = context.interpret()
        self.assertEqual(play_list, [{'Chord': 'Em', 'Tune': '43231323'}])

    def test_interpret_4(self):
        context = AutomaticGuitarSimulator("G63231323")
        play_list = context.interpret()
        self.assertEqual(play_list, [{'Chord': 'G', 'Tune': '63231323'}])

    def test_interpret_5(self):
        context = AutomaticGuitarSimulator("F43231323 G63231323")
        play_list = context.interpret()
        self.assertEqual(play_list, [{'Chord': 'F', 'Tune': '43231323'}, {'Chord': 'G', 'Tune': '63231323'}])

    def test_interpret_6(self):
        context = AutomaticGuitarSimulator(" ")
        play_list = context.interpret()
        self.assertEqual(play_list, [])

    def test_interpret_7(self):
        context = AutomaticGuitarSimulator("ABC43231323 DEF63231323")
        play_list = context.interpret()
        self.assertEqual(play_list, [{'Chord': 'ABC', 'Tune': '43231323'}, {'Chord': 'DEF', 'Tune': '63231323'}])

    def test_interpret_8(self):
        context = AutomaticGuitarSimulator("C53231323")
        play_list = context.interpret(display=True)
        self.assertEqual(play_list, [{'Chord': 'C', 'Tune': '53231323'}])

    def test_interpret_9(self):
        context = AutomaticGuitarSimulator("")
        play_list = context.interpret()
        self.assertEqual(play_list, [])


class AutomaticGuitarSimulatorTestDisplay(unittest.TestCase):
    def test_display_1(self):
        context = AutomaticGuitarSimulator("C53231323 Em43231323")
        play_list = context.interpret()
        str = context.display(play_list[0]['Chord'], play_list[0]['Tune'])
        self.assertEqual(str, "Normal Guitar Playing -- Chord: C, Play Tune: 53231323")

    def test_display_2(self):
        context = AutomaticGuitarSimulator("C53231323 Em43231323")
        play_list = context.interpret()
        str = context.display(play_list[1]['Chord'], play_list[1]['Tune'])
        self.assertEqual(str, "Normal Guitar Playing -- Chord: Em, Play Tune: 43231323")

    def test_display_3(self):
        context = AutomaticGuitarSimulator("F43231323 G63231323")
        play_list = context.interpret()
        str = context.display(play_list[0]['Chord'], play_list[0]['Tune'])
        self.assertEqual(str, "Normal Guitar Playing -- Chord: F, Play Tune: 43231323")

    def test_display_4(self):
        context = AutomaticGuitarSimulator("F43231323 G63231323")
        play_list = context.interpret()
        str = context.display(play_list[1]['Chord'], play_list[1]['Tune'])
        self.assertEqual(str, "Normal Guitar Playing -- Chord: G, Play Tune: 63231323")

    def test_display_5(self):
        context = AutomaticGuitarSimulator("")
        str = context.display('', '')
        self.assertEqual(str, "Normal Guitar Playing -- Chord: , Play Tune: ")


class AutomaticGuitarSimulatorTest(unittest.TestCase):
    def test_AutomaticGuitarSimulator(self):
        context = AutomaticGuitarSimulator("C53231323")
        play_list = context.interpret()
        self.assertEqual(play_list, [{'Chord': 'C', 'Tune': '53231323'}])

        context = AutomaticGuitarSimulator("C53231323 Em43231323")
        play_list = context.interpret()
        str = context.display(play_list[0]['Chord'], play_list[0]['Tune'])
        self.assertEqual(str, "Normal Guitar Playing -- Chord: C, Play Tune: 53231323")
