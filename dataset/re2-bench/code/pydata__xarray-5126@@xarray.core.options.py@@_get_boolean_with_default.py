import warnings
from ..backends.file_manager import FILE_CACHE

ARITHMETIC_JOIN = "arithmetic_join"
CMAP_DIVERGENT = "cmap_divergent"
CMAP_SEQUENTIAL = "cmap_sequential"
DISPLAY_MAX_ROWS = "display_max_rows"
DISPLAY_STYLE = "display_style"
DISPLAY_WIDTH = "display_width"
DISPLAY_EXPAND_ATTRS = "display_expand_attrs"
DISPLAY_EXPAND_COORDS = "display_expand_coords"
DISPLAY_EXPAND_DATA_VARS = "display_expand_data_vars"
DISPLAY_EXPAND_DATA = "display_expand_data"
ENABLE_CFTIMEINDEX = "enable_cftimeindex"
FILE_CACHE_MAXSIZE = "file_cache_maxsize"
KEEP_ATTRS = "keep_attrs"
WARN_FOR_UNCLOSED_FILES = "warn_for_unclosed_files"
OPTIONS = {
    ARITHMETIC_JOIN: "inner",
    CMAP_DIVERGENT: "RdBu_r",
    CMAP_SEQUENTIAL: "viridis",
    DISPLAY_MAX_ROWS: 12,
    DISPLAY_STYLE: "html",
    DISPLAY_WIDTH: 80,
    DISPLAY_EXPAND_ATTRS: "default",
    DISPLAY_EXPAND_COORDS: "default",
    DISPLAY_EXPAND_DATA_VARS: "default",
    DISPLAY_EXPAND_DATA: "default",
    ENABLE_CFTIMEINDEX: True,
    FILE_CACHE_MAXSIZE: 128,
    KEEP_ATTRS: "default",
    WARN_FOR_UNCLOSED_FILES: False,
}
_JOIN_OPTIONS = frozenset(["inner", "outer", "left", "right", "exact"])
_DISPLAY_OPTIONS = frozenset(["text", "html"])
_VALIDATORS = {
    ARITHMETIC_JOIN: _JOIN_OPTIONS.__contains__,
    DISPLAY_MAX_ROWS: _positive_integer,
    DISPLAY_STYLE: _DISPLAY_OPTIONS.__contains__,
    DISPLAY_WIDTH: _positive_integer,
    DISPLAY_EXPAND_ATTRS: lambda choice: choice in [True, False, "default"],
    DISPLAY_EXPAND_COORDS: lambda choice: choice in [True, False, "default"],
    DISPLAY_EXPAND_DATA_VARS: lambda choice: choice in [True, False, "default"],
    DISPLAY_EXPAND_DATA: lambda choice: choice in [True, False, "default"],
    ENABLE_CFTIMEINDEX: lambda value: isinstance(value, bool),
    FILE_CACHE_MAXSIZE: _positive_integer,
    KEEP_ATTRS: lambda choice: choice in [True, False, "default"],
    WARN_FOR_UNCLOSED_FILES: lambda value: isinstance(value, bool),
}
_SETTERS = {
    ENABLE_CFTIMEINDEX: _warn_on_setting_enable_cftimeindex,
    FILE_CACHE_MAXSIZE: _set_file_cache_maxsize,
}

def _get_boolean_with_default(option, default):
    global_choice = OPTIONS[option]

    if global_choice == "default":
        return default
    elif global_choice in [True, False]:
        return global_choice
    else:
        raise ValueError(
            f"The global option {option} must be one of True, False or 'default'."
        )
