

class AutomaticGuitarSimulator():

    def __init__(self, text) -> None:
        self.play_text = text

    def display(self, key, value):
        return ('Normal Guitar Playing -- Chord: %s, Play Tune: %s' % (key, value))
