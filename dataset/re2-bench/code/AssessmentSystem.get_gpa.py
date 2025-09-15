

class AssessmentSystem():

    def __init__(self):
        self.students = {}

    def get_gpa(self, name):
        if ((name in self.students) and self.students[name]['courses']):
            return (sum(self.students[name]['courses'].values()) / len(self.students[name]['courses']))
        else:
            return None
