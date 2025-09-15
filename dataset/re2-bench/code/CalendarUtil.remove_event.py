
from datetime import datetime, timedelta

class CalendarUtil():

    def __init__(self):
        self.events = []

    def remove_event(self, event):
        if (event in self.events):
            self.events.remove(event)
