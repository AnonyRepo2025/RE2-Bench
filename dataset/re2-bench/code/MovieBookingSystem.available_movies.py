
from datetime import datetime
import numpy as np

class MovieBookingSystem():

    def __init__(self):
        self.movies = []

    def available_movies(self, start_time, end_time):
        start_time = datetime.strptime(start_time, '%H:%M')
        end_time = datetime.strptime(end_time, '%H:%M')
        available_movies = []
        for movie in self.movies:
            if ((start_time <= movie['start_time']) and (movie['end_time'] <= end_time)):
                available_movies.append(movie['name'])
        return available_movies
