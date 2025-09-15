
import time

class Thermostat():

    def __init__(self, current_temperature, target_temperature, mode):
        self.current_temperature = current_temperature
        self.target_temperature = target_temperature
        self.mode = mode

    def auto_set_mode(self):
        if (self.current_temperature < self.target_temperature):
            self.mode = 'heat'
        else:
            self.mode = 'cool'

    def simulate_operation(self):
        self.auto_set_mode()
        use_time = 0
        if (self.mode == 'heat'):
            while (self.current_temperature < self.target_temperature):
                self.current_temperature += 1
                use_time += 1
        else:
            while (self.current_temperature > self.target_temperature):
                self.current_temperature -= 1
                use_time += 1
        return use_time
