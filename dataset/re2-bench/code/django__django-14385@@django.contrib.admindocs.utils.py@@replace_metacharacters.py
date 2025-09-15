import re
from email.errors import HeaderParseError
from email.parser import HeaderParser
from inspect import cleandoc
from django.urls import reverse
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
import docutils.core
import docutils.nodes
import docutils.parsers.rst.roles

ROLES = {
    'model': '%s/models/%s/',
    'view': '%s/views/%s/',
    'template': '%s/templates/%s/',
    'filter': '%s/filters/#%s',
    'tag': '%s/tags/#%s',
}
named_group_matcher = _lazy_re_compile(r'\(\?P(<\w+>)')
unnamed_group_matcher = _lazy_re_compile(r'\(')

def replace_metacharacters(pattern):
    """Remove unescaped metacharacters from the pattern."""
    return re.sub(
        r'((?:^|(?<!\\))(?:\\\\)*)(\\?)([?*+^$]|\\[bBAZ])',
        lambda m: m[1] + m[3] if m[2] else m[1],
        pattern,
    )
