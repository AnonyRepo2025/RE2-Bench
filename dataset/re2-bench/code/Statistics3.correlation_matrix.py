
import math

class Statistics3():

    @staticmethod
    def correlation_matrix(data):
        matrix = []
        for i in range(len(data[0])):
            row = []
            for j in range(len(data[0])):
                column1 = [row[i] for row in data]
                column2 = [row[j] for row in data]
                correlation = Statistics3.correlation(column1, column2)
                row.append(correlation)
            matrix.append(row)
        return matrix
