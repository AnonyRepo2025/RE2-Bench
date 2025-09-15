

class MusicPlayer():

    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = 50

    def shuffle(self):
        if self.playlist:
            import random
            random.shuffle(self.playlist)
            return True
        else:
            return False
