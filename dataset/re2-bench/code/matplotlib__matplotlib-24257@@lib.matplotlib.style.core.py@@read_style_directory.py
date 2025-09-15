import contextlib
import logging
import os
from pathlib import Path
import sys
import warnings
import matplotlib as mpl
from matplotlib import _api, _docstring, _rc_params_in_file, rcParamsDefault
import importlib.resources as importlib_resources
import importlib_resources

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

def read_style_directory(style_dir):
    """Return dictionary of styles defined in *style_dir*."""
    styles = dict()
    for path in Path(style_dir).glob(f"*.{STYLE_EXTENSION}"):
        with warnings.catch_warnings(record=True) as warns:
            styles[path.stem] = _rc_params_in_file(path)
        for w in warns:
            _log.warning('In %s: %s', path, w.message)
    return styles
