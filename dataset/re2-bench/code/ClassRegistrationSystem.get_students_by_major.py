

class ClassRegistrationSystem():

    def __init__(self):
        self.students = []
        self.students_registration_classes = {}

    def get_students_by_major(self, major):
        student_list = []
        for student in self.students:
            if (student['major'] == major):
                student_list.append(student['name'])
        return student_list
