

class Hotel():

    def __init__(self, name, rooms):
        self.name = name
        self.available_rooms = rooms
        self.booked_rooms = {}

    def get_available_rooms(self, room_type):
        return self.available_rooms[room_type]
