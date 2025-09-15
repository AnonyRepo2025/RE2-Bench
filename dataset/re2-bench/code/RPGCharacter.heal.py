

class RPGCharacter():

    def __init__(self, name, hp, attack_power, defense, level=1):
        self.name = name
        self.hp = hp
        self.attack_power = attack_power
        self.defense = defense
        self.level = level
        self.exp = 0

    def heal(self):
        self.hp += 10
        if (self.hp > 100):
            self.hp = 100
        return self.hp
