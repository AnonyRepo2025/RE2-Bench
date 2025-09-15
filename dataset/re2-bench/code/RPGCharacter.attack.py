

class RPGCharacter():

    def __init__(self, name, hp, attack_power, defense, level=1):
        self.name = name
        self.hp = hp
        self.attack_power = attack_power
        self.defense = defense
        self.level = level
        self.exp = 0

    def attack(self, other_character):
        damage = max((self.attack_power - other_character.defense), 1)
        other_character.hp -= damage
