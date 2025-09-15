

class WeatherSystem():

    def __init__(self, city) -> None:
        self.temperature = None
        self.weather = None
        self.city = city
        self.weather_list = {}

    def fahrenheit_to_celsius(self):
        return (((self.temperature - 32) * 5) / 9)
