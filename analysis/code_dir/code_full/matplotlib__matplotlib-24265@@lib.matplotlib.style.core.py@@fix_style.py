def warn_deprecated(since, *, message='', name='', alternative='', pending=False, obj_type='', addendum='', removal=''):
    warning = _generate_deprecation_warning(since, message, name, alternative, pending, obj_type, addendum, removal=removal)
    from . import warn_external
    warn_external(warning, category=MatplotlibDeprecationWarning)

def _generate_deprecation_warning(since, message='', name='', alternative='', pending=False, obj_type='', addendum='', *, removal=''):
    if pending:
        if removal:
            raise ValueError('A pending deprecation cannot have a scheduled removal')
    else:
        removal = f'in {removal}' if removal else 'two minor releases later'
    if not message:
        message = ('The %(name)s %(obj_type)s' if obj_type else '%(name)s') + (' will be deprecated in a future version' if pending else ' was deprecated in Matplotlib %(since)s' + (' and will be removed %(removal)s' if removal else '')) + '.' + (' Use %(alternative)s instead.' if alternative else '') + (' %(addendum)s' if addendum else '')
    warning_cls = PendingDeprecationWarning if pending else MatplotlibDeprecationWarning
    return warning_cls(message % dict(func=name, name=name, obj_type=obj_type, since=since, removal=removal, alternative=alternative, addendum=addendum))

def warn_external(message, category=None):
    frame = sys._getframe()
    for stacklevel in itertools.count(1):
        if frame is None:
            break
        if not re.match('\\A(matplotlib|mpl_toolkits)(\\Z|\\.(?!tests\\.))', frame.f_globals.get('__name__', '')):
            break
        frame = frame.f_back
    warnings.warn(message, category, stacklevel)



import contextlib
import logging
import os
from pathlib import Path
import re
import warnings
import matplotlib as mpl
from matplotlib import _api, _docstring, rc_params_from_file, rcParamsDefault
_log = logging.getLogger(__name__)
__all__ = ['use', 'context', 'available', 'library', 'reload_library']
BASE_LIBRARY_PATH = os.path.join(mpl.get_data_path(), 'stylelib')
USER_LIBRARY_PATHS = [os.path.join(mpl.get_configdir(), 'stylelib')]
STYLE_EXTENSION = 'mplstyle'
STYLE_BLACKLIST = {'interactive', 'backend', 'webagg.port', 'webagg.address', 'webagg.port_retries', 'webagg.open_in_browser', 'backend_fallback', 'toolbar', 'timezone', 'figure.max_open_warning', 'figure.raise_window', 'savefig.directory', 'tk.window_focus', 'docstring.hardcopy', 'date.epoch'}
_DEPRECATED_SEABORN_STYLES = {s: s.replace('seaborn', 'seaborn-v0_8') for s in ['seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-dark-palette', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid']}
_DEPRECATED_SEABORN_MSG = "The seaborn styles shipped by Matplotlib are deprecated since %(since)s, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead."
_base_library = read_style_directory(BASE_LIBRARY_PATH)
library = _StyleLibrary()
available = []

def use(style):
    if isinstance(style, (str, Path)) or hasattr(style, 'keys'):
        styles = [style]
    else:
        styles = style
    style_alias = {'mpl20': 'default', 'mpl15': 'classic'}

    def fix_style(s):
        if isinstance(s, str):
            s = style_alias.get(s, s)
            if s in _DEPRECATED_SEABORN_STYLES:
                _api.warn_deprecated('3.6', message=_DEPRECATED_SEABORN_MSG)
                s = _DEPRECATED_SEABORN_STYLES[s]
        return s
    for style in map(fix_style, styles):
        if not isinstance(style, (str, Path)):
            _apply_style(style)
        elif style == 'default':
            with _api.suppress_matplotlib_deprecation_warning():
                _apply_style(rcParamsDefault, warn=False)
        elif style in library:
            _apply_style(library[style])
        else:
            try:
                rc = rc_params_from_file(style, use_default_template=False)
                _apply_style(rc)
            except IOError as err:
                raise IOError('{!r} not found in the style library and input is not a valid URL or path; see `style.available` for list of available styles'.format(style)) from err