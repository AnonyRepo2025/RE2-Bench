
import urllib.parse

class UrlPath():

    def __init__(self):
        self.segments = []
        self.with_end_tag = False

    def add(self, segment):
        self.segments.append(self.fix_path(segment))

    @staticmethod
    def fix_path(path):
        if (not path):
            return ''
        segment_str = path.strip('/')
        return segment_str
