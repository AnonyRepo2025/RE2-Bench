def as_numer_denom(self):
    if not self.is_commutative:
        return self, S.One
    base, exp = self.as_base_exp()
    n, d = base.as_numer_denom()
    # this should be the same as ExpBase.as_numer_denom wrt
    # exponent handling
    neg_exp = exp.is_negative
    if not neg_exp and not (-exp).is_negative:
        neg_exp = _coeff_isneg(exp)
    int_exp = exp.is_integer
    # the denominator cannot be separated from the numerator if
    # its sign is unknown unless the exponent is an integer, e.g.
    # sqrt(a/b) != sqrt(a)/sqrt(b) when a=1 and b=-1. But if the
    # denominator is negative the numerator and denominator can
    # be negated and the denominator (now positive) separated.
    if not (d.is_real or int_exp):
        n = base
        d = S.One
    dnonpos = d.is_nonpositive
    if dnonpos:
        n, d = -n, -d
    elif dnonpos is None and not int_exp:
        n = base
        d = S.One
    if neg_exp:
        n, d = d, n
        exp = -exp
    return self.func(n, exp), self.func(d, exp)

def evalf(x, prec, options):
    from sympy import re as re_, im as im_
    try:
        rf = evalf_table[x.func]
        r = rf(x, prec, options)
    except KeyError:
        try:
            # Fall back to ordinary evalf if possible
            if 'subs' in options:
                x = x.subs(evalf_subs(prec, options['subs']))
            xe = x._eval_evalf(prec)
            re, im = xe.as_real_imag()
            if re.has(re_) or im.has(im_):
                raise NotImplementedError
            if re == 0:
                re = None
                reprec = None
            elif re.is_number:
                re = re._to_mpmath(prec, allow_ints=False)._mpf_
                reprec = prec
            if im == 0:
                im = None
                imprec = None
            elif im.is_number:
                im = im._to_mpmath(prec, allow_ints=False)._mpf_
                imprec = prec
            r = re, im, reprec, imprec
        except AttributeError:
            raise NotImplementedError
    if options.get("verbose"):
        print("### input", x)
        print("### output", to_str(r[0] or fzero, 50))
        print("### raw", r) # r[0], r[2]
        print()
    chop = options.get('chop', False)
    if chop:
        if chop is True:
            chop_prec = prec
        else:
            # convert (approximately) from given tolerance;
            # the formula here will will make 1e-i rounds to 0 for
            # i in the range +/-27 while 2e-i will not be chopped
            chop_prec = int(round(-3.321*math.log10(chop) + 2.5))
            if chop_prec == 3:
                chop_prec -= 1
        r = chop_parts(r, chop_prec)
    if options.get("strict"):
        check_target(x, r, prec)
    return r