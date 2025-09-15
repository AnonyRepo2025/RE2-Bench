

class EightPuzzle():

    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    def find_blank(self, state):
        for i in range(3):
            for j in range(3):
                if (state[i][j] == 0):
                    return (i, j)

    def move(self, state, direction):
        (i, j) = self.find_blank(state)
        new_state = [row[:] for row in state]
        if (direction == 'up'):
            (new_state[i][j], new_state[(i - 1)][j]) = (new_state[(i - 1)][j], new_state[i][j])
        elif (direction == 'down'):
            (new_state[i][j], new_state[(i + 1)][j]) = (new_state[(i + 1)][j], new_state[i][j])
        elif (direction == 'left'):
            (new_state[i][j], new_state[i][(j - 1)]) = (new_state[i][(j - 1)], new_state[i][j])
        elif (direction == 'right'):
            (new_state[i][j], new_state[i][(j + 1)]) = (new_state[i][(j + 1)], new_state[i][j])
        return new_state
