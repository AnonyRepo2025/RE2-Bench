
import json

class CookiesUtil():

    def __init__(self, cookies_file):
        self.cookies_file = cookies_file
        self.cookies = None

    def load_cookies(self):
        try:
            with open(self.cookies_file, 'r') as file:
                cookies_data = json.load(file)
                return cookies_data
        except FileNotFoundError:
            return {}
