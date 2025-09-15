
import datetime
import time

class TimeUtils():

    def __init__(self):
        self.datetime = datetime.datetime.now()

    def get_current_time(self):
        format = '%H:%M:%S'
        return self.datetime.strftime(format)
