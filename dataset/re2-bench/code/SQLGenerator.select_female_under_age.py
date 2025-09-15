

class SQLGenerator():

    def __init__(self, table_name):
        self.table_name = table_name

    def select(self, fields=None, condition=None):
        if (fields is None):
            fields = '*'
        else:
            fields = ', '.join(fields)
        sql = f'SELECT {fields} FROM {self.table_name}'
        if (condition is not None):
            sql += f' WHERE {condition}'
        return (sql + ';')

    def select_female_under_age(self, age):
        condition = f"age < {age} AND gender = 'female'"
        return self.select(condition=condition)
