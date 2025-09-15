

class MusicPlayer():

    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = 50

    def previous_song(self):
        if self.current_song:
            current_index = self.playlist.index(self.current_song)
            if (current_index > 0):
                self.current_song = self.playlist[(current_index - 1)]
                return True
            else:
                return False
        else:
            return False
