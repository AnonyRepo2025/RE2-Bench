

class PersonRequest():

    def __init__(self, name: str, sex: str, phoneNumber: str):
        self.name = self._validate_name(name)
        self.sex = self._validate_sex(sex)
        self.phoneNumber = self._validate_phoneNumber(phoneNumber)

    def _validate_sex(self, sex: str) -> str:
        if (sex not in ['Man', 'Woman', 'UGM']):
            return None
        return sex
