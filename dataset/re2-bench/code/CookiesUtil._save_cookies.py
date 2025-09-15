
import json

class CookiesUtil():

    def __init__(self, cookies_file):
        self.cookies_file = cookies_file
        self.cookies = None

    def _save_cookies(self):
        try:
            with open(self.cookies_file, 'w') as file:
                json.dump(self.cookies, file)
            return True
        except:
            return False
