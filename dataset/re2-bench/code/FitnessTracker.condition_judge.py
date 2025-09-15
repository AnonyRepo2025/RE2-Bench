

class FitnessTracker():

    def __init__(self, height, weight, age, sex) -> None:
        self.height = height
        self.weight = weight
        self.age = age
        self.sex = sex
        self.BMI_std = [{'male': [20, 25]}, {'female': [19, 24]}]

    def get_BMI(self):
        return (self.weight / (self.height ** 2))

    def condition_judge(self):
        BMI = self.get_BMI()
        if (self.sex == 'male'):
            BMI_range = self.BMI_std[0]['male']
        else:
            BMI_range = self.BMI_std[1]['female']
        if (BMI > BMI_range[1]):
            return 1
        elif (BMI < BMI_range[0]):
            return (- 1)
        else:
            return 0
