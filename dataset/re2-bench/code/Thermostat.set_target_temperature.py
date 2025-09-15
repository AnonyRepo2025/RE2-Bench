
import time

class Thermostat():

    def __init__(self, current_temperature, target_temperature, mode):
        self.current_temperature = current_temperature
        self.target_temperature = target_temperature
        self.mode = mode

    def set_target_temperature(self, temperature):
        self.target_temperature = temperature
