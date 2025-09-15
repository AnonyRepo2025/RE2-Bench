

class Hotel():

    def __init__(self, name, rooms):
        self.name = name
        self.available_rooms = rooms
        self.booked_rooms = {}

    def book_room(self, room_type, room_number, name):
        if (room_type not in self.available_rooms.keys()):
            return False
        if (room_number <= self.available_rooms[room_type]):
            if (room_type not in self.booked_rooms.keys()):
                self.booked_rooms[room_type] = {}
            self.booked_rooms[room_type][name] = room_number
            self.available_rooms[room_type] -= room_number
            return 'Success!'
        elif (self.available_rooms[room_type] != 0):
            return self.available_rooms[room_type]
        else:
            return False
