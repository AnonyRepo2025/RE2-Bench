

class HRManagementSystem():

    def __init__(self):
        self.employees = {}

    def add_employee(self, employee_id, name, position, department, salary):
        if (employee_id in self.employees):
            return False
        else:
            self.employees[employee_id] = {'name': name, 'position': position, 'department': department, 'salary': salary}
            return True
