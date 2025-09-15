import inspect
import re
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union
from sphinx.application import Sphinx
from sphinx.config import Config as SphinxConfig
from sphinx.ext.napoleon.iterators import modify_iter
from sphinx.locale import _
from typing import Type  # for python3.5.1
from sphinx.ext.napoleon import Config

_directive_regex = re.compile(r'\.\. \S+::')
_google_section_regex = re.compile(r'^(\s|\w)+:\s*$')
_google_typed_arg_regex = re.compile(r'\s*(.+?)\s*\(\s*(.*[^\s]+)\s*\)')
_numpy_section_regex = re.compile(r'^[=\-`:\'"~^_*+#<>]{2,}\s*$')
_single_colon_regex = re.compile(r'(?<!:):(?!:)')
_xref_or_code_regex = re.compile(
    r'((?::(?:[a-zA-Z0-9]+[\-_+:.])*[a-zA-Z0-9]+:`.+?`)|'
    r'(?:``.+``))')
_bullet_list_regex = re.compile(r'^(\*|\+|\-)(\s+\S|\s*$)')
_enumerated_list_regex = re.compile(
    r'^(?P<paren>\()?'
    r'(\d+|#|[ivxlcdm]+|[IVXLCDM]+|[a-zA-Z])'
    r'(?(paren)\)|\.)(\s+\S|\s*$)')

class GoogleDocstring:
    _name_rgx = re.compile(r"^\s*((?::(?P<role>\S+):)?`(?P<name>~?[a-zA-Z0-9_.-]+)`|"
                           r" (?P<name2>~?[a-zA-Z0-9_.-]+))\s*", re.X)
    def _escape_args_and_kwargs(self, name: str) -> str:
        if name.endswith('_') and getattr(self._config, 'strip_signature_backslash', False):
            name = name[:-1] + r'\_'

        if name[:2] == '**':
            return r'\*\*' + name[2:]
        elif name[:1] == '*':
            return r'\*' + name[1:]
        else:
            return name