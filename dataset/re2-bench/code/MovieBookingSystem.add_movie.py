
from datetime import datetime
import numpy as np

class MovieBookingSystem():

    def __init__(self):
        self.movies = []

    def add_movie(self, name, price, start_time, end_time, n):
        movie = {'name': name, 'price': price, 'start_time': datetime.strptime(start_time, '%H:%M'), 'end_time': datetime.strptime(end_time, '%H:%M'), 'seats': np.zeros((n, n))}
        self.movies.append(movie)
