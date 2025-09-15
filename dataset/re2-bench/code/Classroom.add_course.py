
from datetime import datetime

class Classroom():

    def __init__(self, id):
        self.id = id
        self.courses = []

    def add_course(self, course):
        if (course not in self.courses):
            self.courses.append(course)
