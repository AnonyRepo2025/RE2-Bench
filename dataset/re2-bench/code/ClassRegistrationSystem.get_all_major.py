

class ClassRegistrationSystem():

    def __init__(self):
        self.students = []
        self.students_registration_classes = {}

    def get_all_major(self):
        major_list = []
        for student in self.students:
            if (student['major'] not in major_list):
                major_list.append(student['major'])
        return major_list
