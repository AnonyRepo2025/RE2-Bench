
from datetime import datetime, timedelta

class CalendarUtil():

    def __init__(self):
        self.events = []

    def is_available(self, start_time, end_time):
        for event in self.events:
            if ((start_time < event['end_time']) and (end_time > event['start_time'])):
                return False
        return True
