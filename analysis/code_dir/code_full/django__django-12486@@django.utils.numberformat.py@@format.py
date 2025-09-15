def __getattr__(self, name):
    if self._wrapped is empty:
        self._setup(name)
    val = getattr(self._wrapped, name)
    self.__dict__[name] = val
    return val



from decimal import Decimal
from django.conf import settings
from django.utils.safestring import mark_safe

def format(number, decimal_sep, decimal_pos=None, grouping=0, thousand_sep='', force_grouping=False, use_l10n=None):
    use_grouping = (use_l10n or (use_l10n is None and settings.USE_L10N)) and settings.USE_THOUSAND_SEPARATOR
    use_grouping = use_grouping or force_grouping
    use_grouping = use_grouping and grouping != 0
    if isinstance(number, int) and (not use_grouping) and (not decimal_pos):
        return mark_safe(number)
    sign = ''
    if isinstance(number, float) and 'e' in str(number).lower():
        number = Decimal(str(number))
    if isinstance(number, Decimal):
        if decimal_pos is not None:
            cutoff = Decimal('0.' + '1'.rjust(decimal_pos, '0'))
            if abs(number) < cutoff:
                number = Decimal('0')
        _, digits, exponent = number.as_tuple()
        if abs(exponent) + len(digits) > 200:
            number = '{:e}'.format(number)
            coefficient, exponent = number.split('e')
            coefficient = format(coefficient, decimal_sep, decimal_pos, grouping, thousand_sep, force_grouping, use_l10n)
            return '{}e{}'.format(coefficient, exponent)
        else:
            str_number = '{:f}'.format(number)
    else:
        str_number = str(number)
    if str_number[0] == '-':
        sign = '-'
        str_number = str_number[1:]
    if '.' in str_number:
        int_part, dec_part = str_number.split('.')
        if decimal_pos is not None:
            dec_part = dec_part[:decimal_pos]
    else:
        int_part, dec_part = (str_number, '')
    if decimal_pos is not None:
        dec_part = dec_part + '0' * (decimal_pos - len(dec_part))
    dec_part = dec_part and decimal_sep + dec_part
    if use_grouping:
        try:
            intervals = list(grouping)
        except TypeError:
            intervals = [grouping, 0]
        active_interval = intervals.pop(0)
        int_part_gd = ''
        cnt = 0
        for digit in int_part[::-1]:
            if cnt and cnt == active_interval:
                if intervals:
                    active_interval = intervals.pop(0) or active_interval
                int_part_gd += thousand_sep[::-1]
                cnt = 0
            int_part_gd += digit
            cnt += 1
        int_part = int_part_gd[::-1]
    return sign + int_part + dec_part