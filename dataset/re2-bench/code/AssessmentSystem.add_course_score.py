

class AssessmentSystem():

    def __init__(self):
        self.students = {}

    def add_course_score(self, name, course, score):
        if (name in self.students):
            self.students[name]['courses'][course] = score
