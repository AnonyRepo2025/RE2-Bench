

class FitnessTracker():

    def __init__(self, height, weight, age, sex) -> None:
        self.height = height
        self.weight = weight
        self.age = age
        self.sex = sex
        self.BMI_std = [{'male': [20, 25]}, {'female': [19, 24]}]

    def get_BMI(self):
        return (self.weight / (self.height ** 2))
