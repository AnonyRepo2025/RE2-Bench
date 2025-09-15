

class SQLGenerator():

    def __init__(self, table_name):
        self.table_name = table_name

    def delete(self, condition):
        sql = f'DELETE FROM {self.table_name} WHERE {condition}'
        return (sql + ';')
