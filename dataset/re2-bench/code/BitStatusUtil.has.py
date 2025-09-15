

class BitStatusUtil():

    @staticmethod
    def has(states, stat):
        BitStatusUtil.check([states, stat])
        return ((states & stat) == stat)
