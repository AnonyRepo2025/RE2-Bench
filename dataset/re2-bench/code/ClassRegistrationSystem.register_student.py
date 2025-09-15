

class ClassRegistrationSystem():

    def __init__(self):
        self.students = []
        self.students_registration_classes = {}

    def register_student(self, student):
        if (student in self.students):
            return 0
        else:
            self.students.append(student)
            return 1
