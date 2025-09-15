

class BitStatusUtil():

    @staticmethod
    def add(states, stat):
        BitStatusUtil.check([states, stat])
        return (states | stat)
