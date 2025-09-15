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
       jsonl_path = json_base + "/MusicPlayer.jsonl"
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
# This is a class as a music player that provides to play, stop, add songs, remove songs, set volume, shuffle, and switch to the next or previous song.

class MusicPlayer:
    def __init__(self):
        """
        Initializes the music player with an empty playlist, no current song, and a default volume of 50.
        """
        self.playlist = []
        self.current_song = None
        self.volume = 50

    def add_song(self, song):
        """
        Adds a song to the playlist.
        :param song: The song to add to the playlist, str.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.add_song("song1")
        >>> musicPlayer.playlist
        ['song1']

        """

    def remove_song(self, song):
        """
        Removes a song from the playlist.
        :param song: The song to remove from the playlist, str.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.remove_song("song1")
        >>> musicPlayer.playlist
        ['song2']

        """

    def play(self):
        """
        Plays the current song in the playlist.
        :return: The current song in the playlist, or False if there is no current song.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.current_song = "song1"
        >>> musicPlayer.play()
        'song1'

        """

    def stop(self):
        """
        Stops the current song in the playlist.
        :return: True if the current song was stopped, False if there was no current song.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.current_song = "song1"
        >>> musicPlayer.stop()
        True

        """

    def switch_song(self):
        """
        Switches to the next song in the playlist.
        :return: True if the next song was switched to, False if there was no next song.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.current_song = "song1"
        >>> musicPlayer.switch_song()
        True

        """

    def previous_song(self):
        """
        Switches to the previous song in the playlist.
        :return: True if the previous song was switched to, False if there was no previous song.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.current_song = "song2"
        >>> musicPlayer.previous_song()
        True

        """

    def set_volume(self, volume):
        """
        Sets the volume of the music player,ifthe volume is between 0 and 100 is valid.
        :param volume: The volume to set the music player to,int.
        :return: True if the volume was set, False if the volume was invalid.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.set_volume(50)
        >>> musicPlayer.volume
        50

        """

    def shuffle(self):
        """
        Shuffles the playlist.
        :return: True if the playlist was shuffled, False if the playlist was empty.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.shuffle()
        True

        """
'''
class MusicPlayer:
    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = 50

    @inspect_code
    def add_song(self, song):
        self.playlist.append(song)

    @inspect_code
    def remove_song(self, song):
        if song in self.playlist:
            self.playlist.remove(song)
            if self.current_song == song:
                self.stop()

    @inspect_code
    def play(self):
        if self.playlist and self.current_song:
            return self.playlist[0]
        elif len(self.playlist): 
            return False

    @inspect_code
    def stop(self):
        if self.current_song:
            self.current_song = None
            return True
        else:
            return False

    @inspect_code
    def switch_song(self):
        if self.current_song:
            current_index = self.playlist.index(self.current_song)
            if current_index < len(self.playlist) - 1:
                self.current_song = self.playlist[current_index + 1]
                return True
            else:
                return False
        else:
            return False

    @inspect_code
    def previous_song(self):
        if self.current_song:
            current_index = self.playlist.index(self.current_song)
            if current_index > 0:
                self.current_song = self.playlist[current_index - 1]
                return True
            else:
                return False
        else:
            return False

    @inspect_code
    def set_volume(self, volume):
        if 0 <= volume <= 100:
            self.volume = volume
        else:
            return False

    @inspect_code
    def shuffle(self):
        if self.playlist:
            import random
            random.shuffle(self.playlist)
            return True
        else:
            return False

import unittest


class MusicPlayerTestAddSong(unittest.TestCase):
    def test_add_song(self):
        musicPlayer = MusicPlayer()
        musicPlayer.add_song("song1")
        self.assertEqual(musicPlayer.playlist, ["song1"])

    def test_add_song2(self):
        musicPlayer = MusicPlayer()
        musicPlayer.add_song("song1")
        musicPlayer.add_song("song2")
        self.assertEqual(musicPlayer.playlist, ["song1", "song2"])

    def test_add_song3(self):
        musicPlayer = MusicPlayer()
        musicPlayer.add_song("song1")
        musicPlayer.add_song("song2")
        musicPlayer.add_song("song3")
        self.assertEqual(musicPlayer.playlist, ["song1", "song2", "song3"])

    def test_add_song4(self):
        musicPlayer = MusicPlayer()
        musicPlayer.add_song("song1")
        musicPlayer.add_song("song2")
        musicPlayer.add_song("song3")
        musicPlayer.add_song("song4")
        self.assertEqual(musicPlayer.playlist, ["song1", "song2", "song3", "song4"])

    def test_add_song5(self):
        musicPlayer = MusicPlayer()
        musicPlayer.add_song("song1")
        musicPlayer.add_song("song2")
        musicPlayer.add_song("song3")
        musicPlayer.add_song("song4")
        musicPlayer.add_song("song5")
        self.assertEqual(musicPlayer.playlist, ["song1", "song2", "song3", "song4", "song5"])

class MusicPlayerTestRemoveSong(unittest.TestCase):
    def test_remove_song(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.remove_song("song1")
        self.assertEqual(musicPlayer.playlist, ["song2"])

    def test_remove_song2(self):
        musicPlayer = MusicPlayer()
        musicPlayer.current_song = "song1"
        musicPlayer.playlist = ["song1", "song2", "song3"]
        musicPlayer.remove_song("song1")
        self.assertEqual(musicPlayer.playlist, ["song2", "song3"])

    def test_remove_song3(self):
        musicPlayer = MusicPlayer()
        musicPlayer.current_song = "song1"
        musicPlayer.playlist = ["song1", "song2", "song3", "song4"]
        musicPlayer.remove_song("song1")
        self.assertEqual(musicPlayer.playlist, ["song2", "song3", "song4"])

    def test_remove_song4(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2", "song3", "song4", "song5"]
        musicPlayer.remove_song("song1")
        self.assertEqual(musicPlayer.playlist, ["song2", "song3", "song4", "song5"])

    def test_remove_song5(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2", "song3", "song4", "song5"]
        musicPlayer.remove_song("song1")
        musicPlayer.remove_song("song2")
        self.assertEqual(musicPlayer.playlist, ["song3", "song4", "song5"])

    def test_remove_song6(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = []
        musicPlayer.remove_song("song1")
        self.assertEqual(musicPlayer.playlist, [])


class MusicPlayerTestPlay(unittest.TestCase):
    def test_play(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song1"
        self.assertEqual(musicPlayer.play(), "song1")

    def test_play_2(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = []
        musicPlayer.current_song = "song2"
        self.assertEqual(musicPlayer.play(), None)

    def test_play_3(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        self.assertEqual(musicPlayer.play(),False)

    def test_play_4(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song3"
        self.assertEqual(musicPlayer.play(), "song1")

    def test_play_5(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song1"
        self.assertEqual(musicPlayer.play(), "song1")

class MusicPlayerTestStop(unittest.TestCase):
    def test_stop(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song1"
        self.assertEqual(musicPlayer.stop(), True)

    def test_stop_2(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = []
        musicPlayer.current_song = "song1"
        self.assertEqual(musicPlayer.stop(), True)

    def test_stop_3(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        self.assertEqual(musicPlayer.stop(), False)

    def test_stop_4(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song1"
        self.assertEqual(musicPlayer.stop(), True)

    def test_stop_5(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song2"
        self.assertEqual(musicPlayer.stop(), True)

class MusicPlayerTestSwitchSong(unittest.TestCase):
    def test_switch_song(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song1"
        self.assertEqual(musicPlayer.switch_song(), True)

    def test_switch_song2(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song2"
        self.assertEqual(musicPlayer.switch_song(), False)

    def test_switch_song3(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2", "song3"]
        musicPlayer.current_song = "song3"
        self.assertEqual(musicPlayer.switch_song(), False)

    def test_switch_song4(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        self.assertEqual(musicPlayer.switch_song(), False)

    def test_switch_song5(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = []
        self.assertEqual(musicPlayer.switch_song(), False)

class MusicPlayerTestPreviousSong(unittest.TestCase):
    def test_previous_song(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2", "song3"]
        musicPlayer.current_song = "song2"
        self.assertEqual(musicPlayer.previous_song(), True)

    def test_previous_song2(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2", "song3"]
        musicPlayer.current_song = "song1"
        self.assertEqual(musicPlayer.previous_song(), False)

    def test_previous_song3(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2", "song3"]
        musicPlayer.current_song = "song3"
        self.assertEqual(musicPlayer.previous_song(), True)

    def test_previous_song4(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2", "song3"]
        self.assertEqual(musicPlayer.previous_song(), False)

    def test_previous_song5(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = []
        self.assertEqual(musicPlayer.previous_song(), False)

class MusicPlayerTestSetVolume(unittest.TestCase):
    def test_set_volume(self):
        musicPlayer = MusicPlayer()
        self.assertEqual(musicPlayer.set_volume(50), None)
        self.assertEqual(musicPlayer.volume, 50)

    def test_set_volume2(self):
        musicPlayer = MusicPlayer()
        self.assertEqual(musicPlayer.set_volume(100), None)
        self.assertEqual(musicPlayer.volume, 100)

    def test_set_volume3(self):
        musicPlayer = MusicPlayer()
        self.assertEqual(musicPlayer.set_volume(0), None)
        self.assertEqual(musicPlayer.volume, 0)

    def test_set_volume4(self):
        musicPlayer = MusicPlayer()
        self.assertEqual(musicPlayer.set_volume(101), False)
        self.assertEqual(musicPlayer.volume, 50)

    def test_set_volume5(self):
        musicPlayer = MusicPlayer()
        self.assertEqual(musicPlayer.set_volume(-1), False)
        self.assertEqual(musicPlayer.volume, 50)

class MusicPlayerTestShuffle(unittest.TestCase):
    def test_shuffle(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        self.assertEqual(musicPlayer.shuffle(), True)

    def test_shuffle_2(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = []
        musicPlayer.current_song = "song1"
        self.assertEqual(musicPlayer.shuffle(), False)

    def test_shuffle_3(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song2"
        self.assertEqual(musicPlayer.shuffle(), True)

    def test_shuffle_4(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song3"
        self.assertEqual(musicPlayer.shuffle(), True)

    def test_shuffle_5(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song1"
        self.assertEqual(musicPlayer.shuffle(), True)

class MusicPlayerTestMain(unittest.TestCase):
    def test_main(self):
        musicPlayer = MusicPlayer()
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song1"
        self.assertEqual(musicPlayer.play(), "song1")
        self.assertEqual(musicPlayer.stop(), True)
        musicPlayer.playlist = ["song1", "song2"]
        musicPlayer.current_song = "song1"
        self.assertEqual(musicPlayer.switch_song(), True)
        self.assertEqual(musicPlayer.previous_song(), True)
        musicPlayer.set_volume(50)
        self.assertEqual(musicPlayer.volume, 50)

