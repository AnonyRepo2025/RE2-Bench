import calendar
import datetime
from email.utils import format_datetime as format_datetime_rfc5322
from django.utils.dates import (
    MONTHS, MONTHS_3, MONTHS_ALT, MONTHS_AP, WEEKDAYS, WEEKDAYS_ABBR,
)
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import (
    _datetime_ambiguous_or_imaginary, get_default_timezone, is_naive,
    make_aware,
)
from django.utils.translation import gettext as _

re_formatchars = _lazy_re_compile(r'(?<!\\)([aAbcdDeEfFgGhHiIjlLmMnNoOPrsStTUuwWyYzZ])')
re_escaped = _lazy_re_compile(r'\\(.)')

class DateFormat(TimeFormat):
    def U(self):
        "Seconds since the Unix epoch (January 1 1970 00:00:00 GMT)"
        value = self.data
        if not isinstance(value, datetime.datetime):
            value = datetime.datetime.combine(value, datetime.time.min)
        return int(value.timestamp())