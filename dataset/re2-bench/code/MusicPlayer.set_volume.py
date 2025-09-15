

class MusicPlayer():

    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = 50

    def set_volume(self, volume):
        if (0 <= volume <= 100):
            self.volume = volume
        else:
            return False
