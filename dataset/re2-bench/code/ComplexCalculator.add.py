

class ComplexCalculator():

    def __init__(self):
        pass

    @staticmethod
    def add(c1, c2):
        real = (c1.real + c2.real)
        imaginary = (c1.imag + c2.imag)
        answer = complex(real, imaginary)
        return answer
