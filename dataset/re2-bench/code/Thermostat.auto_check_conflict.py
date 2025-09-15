
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

    def auto_check_conflict(self):
        if (self.current_temperature > self.target_temperature):
            if (self.mode == 'cool'):
                return True
            else:
                self.auto_set_mode()
                return False
        elif (self.mode == 'heat'):
            return True
        else:
            self.auto_set_mode()
            return False
