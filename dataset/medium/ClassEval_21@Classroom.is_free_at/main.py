import unittest
from datetime import datetime


class Classroom:
    def __init__(self, id):
        self.id = id
        self.courses = []

    def add_course(self, course):

        if course not in self.courses:
            self.courses.append(course)

    def remove_course(self, course):
        if course in self.courses:
            self.courses.remove(course)

    def is_free_at(self, check_time):
        check_time = datetime.strptime(check_time, '%H:%M')

        for course in self.courses:
            if datetime.strptime(course['start_time'], '%H:%M') <= check_time <= datetime.strptime(course['end_time'],
                                                                                                   '%H:%M'):
                return False
        return True

    def check_course_conflict(self, new_course):
        new_start_time = datetime.strptime(new_course['start_time'], '%H:%M')
        new_end_time = datetime.strptime(new_course['end_time'], '%H:%M')

        flag = True
        for course in self.courses:
            start_time = datetime.strptime(course['start_time'], '%H:%M')
            end_time = datetime.strptime(course['end_time'], '%H:%M')
            if start_time <= new_start_time and end_time >= new_start_time:
                flag = False
            if start_time <= new_end_time and end_time >= new_end_time:
                flag = False
        return flag
class Test(unittest.TestCase):
    def test(self, check_time):
            classroom = Classroom(1)
            course = {'name': 'math', 'start_time': '09:00', 'end_time': '10:00'}
            classroom.add_course(course)
            check_time = check_time
            result = classroom.is_free_at(check_time)
            return result