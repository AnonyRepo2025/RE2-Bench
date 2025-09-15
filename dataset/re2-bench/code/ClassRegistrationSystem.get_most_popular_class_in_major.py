

class ClassRegistrationSystem():

    def __init__(self):
        self.students = []
        self.students_registration_classes = {}

    def get_most_popular_class_in_major(self, major):
        class_list = []
        for student in self.students:
            if (student['major'] == major):
                class_list += self.students_registration_classes[student['name']]
        most_popular_class = max(set(class_list), key=class_list.count)
        return most_popular_class
