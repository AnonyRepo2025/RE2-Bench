
from datetime import datetime
import numpy as np

class MovieBookingSystem():

    def __init__(self):
        self.movies = []

    def book_ticket(self, name, seats_to_book):
        for movie in self.movies:
            if (movie['name'] == name):
                for seat in seats_to_book:
                    if (movie['seats'][seat[0]][seat[1]] == 0):
                        movie['seats'][seat[0]][seat[1]] = 1
                    else:
                        return 'Booking failed.'
                return 'Booking success.'
        return 'Movie not found.'
