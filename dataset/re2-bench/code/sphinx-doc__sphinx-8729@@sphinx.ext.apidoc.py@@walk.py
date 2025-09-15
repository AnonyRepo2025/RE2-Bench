import argparse
import glob
import locale
import os
import sys
import warnings
from copy import copy
from fnmatch import fnmatch
from importlib.machinery import EXTENSION_SUFFIXES
from os import path
from typing import Any, Generator, List, Tuple
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.cmd.quickstart import EXTENSIONS
from sphinx.deprecation import RemovedInSphinx40Warning, deprecated_alias
from sphinx.locale import __
from sphinx.util import rst
from sphinx.util.osutil import FileAvoidWrite, ensuredir
from sphinx.util.template import ReSTRenderer
from sphinx.cmd import quickstart as qs

PY_SUFFIXES = ('.py', '.pyx') + tuple(EXTENSION_SUFFIXES)
template_dir = path.join(package_dir, 'templates', 'apidoc')

def walk(rootpath: str, excludes: List[str], opts: Any
         ) -> Generator[Tuple[str, List[str], List[str]], None, None]:
    """Walk through the directory and list files and subdirectories up."""
    followlinks = getattr(opts, 'followlinks', False)
    includeprivate = getattr(opts, 'includeprivate', False)

    for root, subs, files in os.walk(rootpath, followlinks=followlinks):
        # document only Python module files (that aren't excluded)
        files = sorted(f for f in files
                       if f.endswith(PY_SUFFIXES) and
                       not is_excluded(path.join(root, f), excludes))

        # remove hidden ('.') and private ('_') directories, as well as
        # excluded dirs
        if includeprivate:
            exclude_prefixes = ('.',)  # type: Tuple[str, ...]
        else:
            exclude_prefixes = ('.', '_')

        subs[:] = sorted(sub for sub in subs if not sub.startswith(exclude_prefixes) and
                         not is_excluded(path.join(root, sub), excludes))

        yield root, subs, files
