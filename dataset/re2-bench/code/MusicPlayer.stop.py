

class MusicPlayer():

    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = 50

    def stop(self):
        if self.current_song:
            self.current_song = None
            return True
        else:
            return False
