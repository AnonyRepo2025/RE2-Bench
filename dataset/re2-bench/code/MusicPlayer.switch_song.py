

class MusicPlayer():

    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = 50

    def switch_song(self):
        if self.current_song:
            current_index = self.playlist.index(self.current_song)
            if (current_index < (len(self.playlist) - 1)):
                self.current_song = self.playlist[(current_index + 1)]
                return True
            else:
                return False
        else:
            return False
