
from datetime import datetime, timedelta

class CalendarUtil():

    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)
