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
STYLE_BLACKLIST = {
    'interactive', 'backend', 'webagg.port', 'webagg.address',
    'webagg.port_retries', 'webagg.open_in_browser', 'backend_fallback',
    'toolbar', 'timezone', 'figure.max_open_warning',
    'figure.raise_window', 'savefig.directory', 'tk.window_focus',
    'docstring.hardcopy', 'date.epoch'}
_DEPRECATED_SEABORN_STYLES = {
    s: s.replace("seaborn", "seaborn-v0_8")
    for s in [
        "seaborn",
        "seaborn-bright",
        "seaborn-colorblind",
        "seaborn-dark",
        "seaborn-darkgrid",
        "seaborn-dark-palette",
        "seaborn-deep",
        "seaborn-muted",
        "seaborn-notebook",
        "seaborn-paper",
        "seaborn-pastel",
        "seaborn-poster",
        "seaborn-talk",
        "seaborn-ticks",
        "seaborn-white",
        "seaborn-whitegrid",
    ]
}
_DEPRECATED_SEABORN_MSG = (
    "The seaborn styles shipped by Matplotlib are deprecated since %(since)s, "
    "as they no longer correspond to the styles shipped by seaborn. However, "
    "they will remain available as 'seaborn-v0_8-<style>'. Alternatively, "
    "directly use the seaborn API instead.")
_base_library = read_style_directory(BASE_LIBRARY_PATH)
library = _StyleLibrary()
available = []

def use(style):
    """
    Use Matplotlib style settings from a style specification.

    The style name of 'default' is reserved for reverting back to
    the default style settings.

    .. note::

       This updates the `.rcParams` with the settings from the style.
       `.rcParams` not defined in the style are kept.

    Parameters
    ----------
    style : str, dict, Path or list
        A style specification. Valid options are:

        +------+-------------------------------------------------------------+
        | str  | The name of a style or a path/URL to a style file. For a    |
        |      | list of available style names, see `.style.available`.      |
        +------+-------------------------------------------------------------+
        | dict | Dictionary with valid key/value pairs for                   |
        |      | `matplotlib.rcParams`.                                      |
        +------+-------------------------------------------------------------+
        | Path | A path-like object which is a path to a style file.         |
        +------+-------------------------------------------------------------+
        | list | A list of style specifiers (str, Path or dict) applied from |
        |      | first to last in the list.                                  |
        +------+-------------------------------------------------------------+

    Notes
    -----
    The following `.rcParams` are not related to style and will be ignored if
    found in a style specification:

    %s
    """
    if isinstance(style, (str, Path)) or hasattr(style, 'keys'):
        # If name is a single str, Path or dict, make it a single element list.
        styles = [style]
    else:
        styles = style

    style_alias = {'mpl20': 'default', 'mpl15': 'classic'}

    def fix_style(s):
        if isinstance(s, str):
            s = style_alias.get(s, s)
            if s in _DEPRECATED_SEABORN_STYLES:
                _api.warn_deprecated("3.6", message=_DEPRECATED_SEABORN_MSG)
                s = _DEPRECATED_SEABORN_STYLES[s]
        return s

    for style in map(fix_style, styles):
        if not isinstance(style, (str, Path)):
            _apply_style(style)
        elif style == 'default':
            # Deprecation warnings were already handled when creating
            # rcParamsDefault, no need to reemit them here.
            with _api.suppress_matplotlib_deprecation_warning():
                _apply_style(rcParamsDefault, warn=False)
        elif style in library:
            _apply_style(library[style])
        else:
            try:
                rc = rc_params_from_file(style, use_default_template=False)
                _apply_style(rc)
            except IOError as err:
                raise IOError(
                    "{!r} not found in the style library and input is not a "
                    "valid URL or path; see `style.available` for list of "
                    "available styles".format(style)) from err
