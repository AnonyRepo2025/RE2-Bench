import os
import re
import warnings
from datetime import datetime, timezone
from os import path
from typing import TYPE_CHECKING, Callable, Generator, List, NamedTuple, Tuple, Union
import babel.dates
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.osutil import SEP, canon_path, relpath
from sphinx.environment import BuildEnvironment

logger = logging.getLogger(__name__)
date_format_mappings = {
    '%a':  'EEE',     # Weekday as locale’s abbreviated name.
    '%A':  'EEEE',    # Weekday as locale’s full name.
    '%b':  'MMM',     # Month as locale’s abbreviated name.
    '%B':  'MMMM',    # Month as locale’s full name.
    '%c':  'medium',  # Locale’s appropriate date and time representation.
    '%-d': 'd',       # Day of the month as a decimal number.
    '%d':  'dd',      # Day of the month as a zero-padded decimal number.
    '%-H': 'H',       # Hour (24-hour clock) as a decimal number [0,23].
    '%H':  'HH',      # Hour (24-hour clock) as a zero-padded decimal number [00,23].
    '%-I': 'h',       # Hour (12-hour clock) as a decimal number [1,12].
    '%I':  'hh',      # Hour (12-hour clock) as a zero-padded decimal number [01,12].
    '%-j': 'D',       # Day of the year as a decimal number.
    '%j':  'DDD',     # Day of the year as a zero-padded decimal number.
    '%-m': 'M',       # Month as a decimal number.
    '%m':  'MM',      # Month as a zero-padded decimal number.
    '%-M': 'm',       # Minute as a decimal number [0,59].
    '%M':  'mm',      # Minute as a zero-padded decimal number [00,59].
    '%p':  'a',       # Locale’s equivalent of either AM or PM.
    '%-S': 's',       # Second as a decimal number.
    '%S':  'ss',      # Second as a zero-padded decimal number.
    '%U':  'WW',      # Week number of the year (Sunday as the first day of the week)
                      # as a zero padded decimal number. All days in a new year preceding
                      # the first Sunday are considered to be in week 0.
    '%w':  'e',       # Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.
    '%-W': 'W',       # Week number of the year (Monday as the first day of the week)
                      # as a decimal number. All days in a new year preceding the first
                      # Monday are considered to be in week 0.
    '%W':  'WW',      # Week number of the year (Monday as the first day of the week)
                      # as a zero-padded decimal number.
    '%x':  'medium',  # Locale’s appropriate date representation.
    '%X':  'medium',  # Locale’s appropriate time representation.
    '%y':  'YY',      # Year without century as a zero-padded decimal number.
    '%Y':  'yyyy',    # Year with century as a decimal number.
    '%Z':  'zzz',     # Time zone name (no characters if no time zone exists).
    '%z':  'ZZZ',     # UTC offset in the form ±HHMM[SS[.ffffff]]
                      # (empty string if the object is naive).
    '%%':  '%',
}
date_format_re = re.compile('(%s)' % '|'.join(date_format_mappings))

def format_date(format: str, date: datetime = None, language: str = None) -> str:
    if date is None:
        # If time is not specified, try to use $SOURCE_DATE_EPOCH variable
        # See https://wiki.debian.org/ReproducibleBuilds/TimestampsProposal
        source_date_epoch = os.getenv('SOURCE_DATE_EPOCH')
        if source_date_epoch is not None:
            date = datetime.utcfromtimestamp(float(source_date_epoch))
        else:
            date = datetime.now(timezone.utc).astimezone()

    if language is None:
        warnings.warn('The language argument for format_date() becomes required.',
                      RemovedInSphinx70Warning)
        language = 'en'

    result = []
    tokens = date_format_re.split(format)
    for token in tokens:
        if token in date_format_mappings:
            babel_format = date_format_mappings.get(token, '')

            # Check if we have to use a different babel formatter then
            # format_datetime, because we only want to format a date
            # or a time.
            if token == '%x':
                function = babel.dates.format_date
            elif token == '%X':
                function = babel.dates.format_time
            else:
                function = babel.dates.format_datetime

            result.append(babel_format_date(date, babel_format, locale=language,
                                            formatter=function))
        else:
            result.append(token)

    return "".join(result)
