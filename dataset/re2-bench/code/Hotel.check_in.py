

class Hotel():

    def __init__(self, name, rooms):
        self.name = name
        self.available_rooms = rooms
        self.booked_rooms = {}

    def check_in(self, room_type, room_number, name):
        if (room_type not in self.booked_rooms.keys()):
            return False
        if (name in self.booked_rooms[room_type]):
            if (room_number > self.booked_rooms[room_type][name]):
                return False
            elif (room_number == self.booked_rooms[room_type][name]):
                self.booked_rooms[room_type].pop(name)
            else:
                self.booked_rooms[room_type][name] -= room_number
