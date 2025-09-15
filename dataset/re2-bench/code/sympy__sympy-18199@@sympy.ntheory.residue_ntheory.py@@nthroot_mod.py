from __future__ import print_function, division
from sympy.core.compatibility import as_int, range
from sympy.core.function import Function
from sympy.utilities.iterables import cartes
from sympy.core.numbers import igcd, igcdex, mod_inverse
from sympy.core.power import isqrt
from sympy.core.singleton import S
from .primetest import isprime
from .factor_ import factorint, trailing, totient, multiplicity
from random import randint, Random
from collections import defaultdict
import itertools
from sympy.polys.galoistools import gf_crt1, gf_crt2
from sympy.polys.domains import ZZ
from sympy.core.numbers import igcdex
from sympy.polys.domains import ZZ
from sympy.ntheory.modular import crt
from sympy.core.numbers import igcdex
from .modular import crt



def nthroot_mod(a, n, p, all_roots=False):
    """
    Find the solutions to ``x**n = a mod p``

    Parameters
    ==========

    a : integer
    n : positive integer
    p : positive integer
    all_roots : if False returns the smallest root, else the list of roots

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import nthroot_mod
    >>> nthroot_mod(11, 4, 19)
    8
    >>> nthroot_mod(11, 4, 19, True)
    [8, 11]
    >>> nthroot_mod(68, 3, 109)
    23
    """
    from sympy.core.numbers import igcdex
    a, n, p = as_int(a), as_int(n), as_int(p)
    if n == 2:
        return sqrt_mod(a, p, all_roots)
    # see Hackman "Elementary Number Theory" (2009), page 76
    if not isprime(p):
        return _nthroot_mod_composite(a, n, p)
    if a % p == 0:
        return [0]
    if not is_nthpow_residue(a, n, p):
        return None
    if (p - 1) % n == 0:
        return _nthroot_mod1(a, n, p, all_roots)
    # The roots of ``x**n - a = 0 (mod p)`` are roots of
    # ``gcd(x**n - a, x**(p - 1) - 1) = 0 (mod p)``
    pa = n
    pb = p - 1
    b = 1
    if pa < pb:
        a, pa, b, pb = b, pb, a, pa
    while pb:
        # x**pa - a = 0; x**pb - b = 0
        # x**pa - a = x**(q*pb + r) - a = (x**pb)**q * x**r - a =
        #             b**q * x**r - a; x**r - c = 0; c = b**-q * a mod p
        q, r = divmod(pa, pb)
        c = pow(b, q, p)
        c = igcdex(c, p)[0]
        c = (c * a) % p
        pa, pb = pb, r
        a, b = b, c
    if pa == 1:
        if all_roots:
            res = [a]
        else:
            res = a
    elif pa == 2:
        return sqrt_mod(a, p , all_roots)
    else:
        res = _nthroot_mod1(a, pa, p, all_roots)
    return res
