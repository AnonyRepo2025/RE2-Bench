

class PersonRequest():

    def __init__(self, name: str, sex: str, phoneNumber: str):
        self.name = self._validate_name(name)
        self.sex = self._validate_sex(sex)
        self.phoneNumber = self._validate_phoneNumber(phoneNumber)

    def _validate_phoneNumber(self, phoneNumber: str) -> str:
        if (not phoneNumber):
            return None
        if ((len(phoneNumber) != 11) or (not phoneNumber.isdigit())):
            return None
        return phoneNumber
