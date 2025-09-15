

class BitStatusUtil():

    @staticmethod
    def remove(states, stat):
        BitStatusUtil.check([states, stat])
        if BitStatusUtil.has(states, stat):
            return (states ^ stat)
        return states
