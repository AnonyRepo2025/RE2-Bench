from sympy.core import S
from sympy.core.relational import Eq, Ne
from sympy.logic.boolalg import BooleanFunction
from sympy.utilities.misc import func_name
from .sets import Set



class Contains(BooleanFunction):
    def as_set(self):
        return self.args[1]