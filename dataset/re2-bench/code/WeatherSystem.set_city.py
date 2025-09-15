

class WeatherSystem():

    def __init__(self, city) -> None:
        self.temperature = None
        self.weather = None
        self.city = city
        self.weather_list = {}

    def set_city(self, city):
        self.city = city
