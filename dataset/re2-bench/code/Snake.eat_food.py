
import random

class Snake():

    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, food_position):
        self.length = 1
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.BLOCK_SIZE = BLOCK_SIZE
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.score = 0
        self.food_position = food_position

    def random_food_position(self):
        while (self.food_position in self.positions):
            self.food_position = ((random.randint(0, ((self.SCREEN_WIDTH // self.BLOCK_SIZE) - 1)) * self.BLOCK_SIZE), (random.randint(0, ((self.SCREEN_HEIGHT // self.BLOCK_SIZE) - 1)) * self.BLOCK_SIZE))

    def eat_food(self):
        self.length += 1
        self.score += 100
        self.random_food_position()
