

class HRManagementSystem():

    def __init__(self):
        self.employees = {}

    def get_employee(self, employee_id):
        if (employee_id in self.employees):
            return self.employees[employee_id]
        else:
            return False
