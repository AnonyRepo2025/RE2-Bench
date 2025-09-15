from sympy.core import sympify, SympifyError
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyutils import PicklableWithSlots
from sympy.utilities import public
from sympy.polys import gcd
from sympy.polys import lcm

eflags = dict(deep=False, mul=True, power_exp=False, power_base=False,
              basic=False, multinomial=False, log=False)

class Expression(PicklableWithSlots):