

class Hotel():

    def __init__(self, name, rooms):
        self.name = name
        self.available_rooms = rooms
        self.booked_rooms = {}

    def check_out(self, room_type, room_number):
        if (room_type in self.available_rooms):
            self.available_rooms[room_type] += room_number
        else:
            self.available_rooms[room_type] = room_number
