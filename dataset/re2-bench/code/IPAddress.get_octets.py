

class IPAddress():

    def __init__(self, ip_address):
        self.ip_address = ip_address

    def is_valid(self):
        octets = self.ip_address.split('.')
        if (len(octets) != 4):
            return False
        for octet in octets:
            if ((not octet.isdigit()) or (int(octet) < 0) or (int(octet) > 255)):
                return False
        return True

    def get_octets(self):
        if self.is_valid():
            return self.ip_address.split('.')
        else:
            return []
