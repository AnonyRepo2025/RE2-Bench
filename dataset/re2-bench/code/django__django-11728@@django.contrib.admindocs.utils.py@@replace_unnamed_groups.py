import re
from email.errors import HeaderParseError
from email.parser import HeaderParser
from django.urls import reverse
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
named_group_matcher = re.compile(r'\(\?P(<\w+>)')
unnamed_group_matcher = re.compile(r'\(')

def replace_unnamed_groups(pattern):
    r"""
    Find unnamed groups in `pattern` and replace them with '<var>'. E.g.,
    1. ^(?P<a>\w+)/b/(\w+)$ ==> ^(?P<a>\w+)/b/<var>$
    2. ^(?P<a>\w+)/b/((x|y)\w+)$ ==> ^(?P<a>\w+)/b/<var>$
    3. ^(?P<a>\w+)/b/(\w+) ==> ^(?P<a>\w+)/b/<var>
    4. ^(?P<a>\w+)/b/((x|y)\w+) ==> ^(?P<a>\w+)/b/<var>
    """
    unnamed_group_indices = [m.start(0) for m in unnamed_group_matcher.finditer(pattern)]
    # Indices of the start of unnamed capture groups.
    group_indices = []
    # Loop over the start indices of the groups.
    for start in unnamed_group_indices:
        # Handle nested parentheses, e.g. '^b/((x|y)\w+)$'.
        unmatched_open_brackets, prev_char = 1, None
        for idx, val in enumerate(pattern[start + 1:]):
            # Check for unescaped `(` and `)`. They mark the start and end of
            # a nested group.
            if val == '(' and prev_char != '\\':
                unmatched_open_brackets += 1
            elif val == ')' and prev_char != '\\':
                unmatched_open_brackets -= 1
            prev_char = val

            if unmatched_open_brackets == 0:
                group_indices.append((start, start + 2 + idx))
                break
    # Remove unnamed group matches inside other unnamed capture groups.
    group_start_end_indices = []
    prev_end = None
    for start, end in group_indices:
        if prev_end and start > prev_end or not prev_end:
            group_start_end_indices.append((start, end))
        prev_end = end

    if group_start_end_indices:
        # Replace unnamed groups with <var>. Handle the fact that replacing the
        # string between indices will change string length and thus indices
        # will point to the wrong substring if not corrected.
        final_pattern, prev_end = [], None
        for start, end in group_start_end_indices:
            if prev_end:
                final_pattern.append(pattern[prev_end:start])
            final_pattern.append(pattern[:start] + '<var>')
            prev_end = end
        final_pattern.append(pattern[prev_end:])
        return ''.join(final_pattern)
    else:
        return pattern
