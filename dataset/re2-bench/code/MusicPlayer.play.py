

class MusicPlayer():

    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = 50

    def play(self):
        if (self.playlist and self.current_song):
            return self.playlist[0]
        elif len(self.playlist):
            return False
