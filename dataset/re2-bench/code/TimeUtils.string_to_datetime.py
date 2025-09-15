
import datetime
import time

class TimeUtils():

    def __init__(self):
        self.datetime = datetime.datetime.now()

    def string_to_datetime(self, string):
        return datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
