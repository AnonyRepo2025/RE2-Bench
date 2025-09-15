
from datetime import datetime

class Classroom():

    def __init__(self, id):
        self.id = id
        self.courses = []

    def is_free_at(self, check_time):
        check_time = datetime.strptime(check_time, '%H:%M')
        for course in self.courses:
            if (datetime.strptime(course['start_time'], '%H:%M') <= check_time <= datetime.strptime(course['end_time'], '%H:%M')):
                return False
        return True
