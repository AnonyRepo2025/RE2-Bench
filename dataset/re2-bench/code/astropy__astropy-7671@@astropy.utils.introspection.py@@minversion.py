import inspect
import re
import types
import importlib
from distutils.version import LooseVersion

__all__ = ['resolve_name', 'minversion', 'find_current_module',
           'isinstancemethod']
__doctest_skip__ = ['find_current_module']

def minversion(module, version, inclusive=True, version_path='__version__'):
    """
    Returns `True` if the specified Python module satisfies a minimum version
    requirement, and `False` if not.

    Parameters
    ----------

    module : module or `str`
        An imported module of which to check the version, or the name of
        that module (in which case an import of that module is attempted--
        if this fails `False` is returned).

    version : `str`
        The version as a string that this module must have at a minimum (e.g.
        ``'0.12'``).

    inclusive : `bool`
        The specified version meets the requirement inclusively (i.e. ``>=``)
        as opposed to strictly greater than (default: `True`).

    version_path : `str`
        A dotted attribute path to follow in the module for the version.
        Defaults to just ``'__version__'``, which should work for most Python
        modules.

    Examples
    --------

    >>> import astropy
    >>> minversion(astropy, '0.4.4')
    True
    """
    if isinstance(module, types.ModuleType):
        module_name = module.__name__
    elif isinstance(module, str):
        module_name = module
        try:
            module = resolve_name(module_name)
        except ImportError:
            return False
    else:
        raise ValueError('module argument must be an actual imported '
                         'module, or the import name of the module; '
                         'got {0!r}'.format(module))

    if '.' not in version_path:
        have_version = getattr(module, version_path)
    else:
        have_version = resolve_name(module.__name__, version_path)

    # LooseVersion raises a TypeError when strings like dev, rc1 are part
    # of the version number. Match the dotted numbers only. Regex taken
    # from PEP440, https://www.python.org/dev/peps/pep-0440/, Appendix B
    expr = '^([1-9]\\d*!)?(0|[1-9]\\d*)(\\.(0|[1-9]\\d*))*'
    m = re.match(expr, version)
    if m:
        version = m.group(0)

    if inclusive:
        return LooseVersion(have_version) >= LooseVersion(version)
    else:
        return LooseVersion(have_version) > LooseVersion(version)
