

import re
from email.errors import HeaderParseError
from email.parser import HeaderParser
from django.urls import reverse
from django.utils.safestring import mark_safe
import docutils.core
import docutils.nodes
import docutils.parsers.rst.roles
ROLES = {'model': '%s/models/%s/', 'view': '%s/views/%s/', 'template': '%s/templates/%s/', 'filter': '%s/filters/#%s', 'tag': '%s/tags/#%s'}
named_group_matcher = re.compile('\\(\\?P(<\\w+>)')
unnamed_group_matcher = re.compile('\\(')

def replace_named_groups(pattern):
    named_group_indices = [(m.start(0), m.end(0), m.group(1)) for m in named_group_matcher.finditer(pattern)]
    group_pattern_and_name = []
    for start, end, group_name in named_group_indices:
        unmatched_open_brackets, prev_char = (1, None)
        for idx, val in enumerate(pattern[end:]):
            if val == '(' and prev_char != '\\':
                unmatched_open_brackets += 1
            elif val == ')' and prev_char != '\\':
                unmatched_open_brackets -= 1
            prev_char = val
            if unmatched_open_brackets == 0:
                group_pattern_and_name.append((pattern[start:end + idx + 1], group_name))
                break
    for group_pattern, group_name in group_pattern_and_name:
        pattern = pattern.replace(group_pattern, group_name)
    return pattern