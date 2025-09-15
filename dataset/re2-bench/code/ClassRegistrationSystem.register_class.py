

class ClassRegistrationSystem():

    def __init__(self):
        self.students = []
        self.students_registration_classes = {}

    def register_class(self, student_name, class_name):
        if (student_name in self.students_registration_classes):
            self.students_registration_classes[student_name].append(class_name)
        else:
            self.students_registration_classes[student_name] = [class_name]
        return self.students_registration_classes[student_name]
