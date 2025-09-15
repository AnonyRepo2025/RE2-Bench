

from __future__ import print_function, division
import re
import collections
_name_with_digits_p = re.compile('^([a-zA-Z]+)([0-9]+)$')

def split_super_sub(text):
    if len(text) == 0:
        return (text, [], [])
    pos = 0
    name = None
    supers = []
    subs = []
    while pos < len(text):
        start = pos + 1
        if text[pos:pos + 2] == '__':
            start += 1
        pos_hat = text.find('^', start)
        if pos_hat < 0:
            pos_hat = len(text)
        pos_usc = text.find('_', start)
        if pos_usc < 0:
            pos_usc = len(text)
        pos_next = min(pos_hat, pos_usc)
        part = text[pos:pos_next]
        pos = pos_next
        if name is None:
            name = part
        elif part.startswith('^'):
            supers.append(part[1:])
        elif part.startswith('__'):
            supers.append(part[2:])
        elif part.startswith('_'):
            subs.append(part[1:])
        else:
            raise RuntimeError('This should never happen.')
    m = _name_with_digits_p.match(name)
    if m:
        name, sub = m.groups()
        subs.insert(0, sub)
    return (name, supers, subs)