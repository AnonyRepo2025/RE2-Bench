
from datetime import datetime, timedelta

class CalendarUtil():

    def __init__(self):
        self.events = []

    def get_events(self, date):
        events_on_date = []
        for event in self.events:
            if (event['date'].date() == date.date()):
                events_on_date.append(event)
        return events_on_date
