

class HRManagementSystem():

    def __init__(self):
        self.employees = {}

    def remove_employee(self, employee_id):
        if (employee_id in self.employees):
            del self.employees[employee_id]
            return True
        else:
            return False
