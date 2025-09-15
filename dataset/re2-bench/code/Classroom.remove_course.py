
from datetime import datetime

class Classroom():

    def __init__(self, id):
        self.id = id
        self.courses = []

    def remove_course(self, course):
        if (course in self.courses):
            self.courses.remove(course)
