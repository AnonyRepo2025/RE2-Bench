import ast
from functools import lru_cache, reduce
from numbers import Real
import operator
import os
import re
import numpy as np
from matplotlib import _api, cbook
from matplotlib.cbook import ls_mapper
from matplotlib.colors import Colormap, is_color_like
from matplotlib._fontconfig_pattern import parse_fontconfig_pattern
from matplotlib._enums import JoinStyle, CapStyle
from cycler import Cycler, cycler as ccycler

interactive_bk = [
    'GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo',
    'MacOSX',
    'nbAgg',
    'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo',
    'TkAgg', 'TkCairo',
    'WebAgg',
    'WX', 'WXAgg', 'WXCairo',
]
non_interactive_bk = ['agg', 'cairo',
                      'pdf', 'pgf', 'ps', 'svg', 'template']
all_backends = interactive_bk + non_interactive_bk
validate_anylist = _listify_validator(validate_any)
validate_string = _make_type_validator(str)
validate_string_or_None = _make_type_validator(str, allow_none=True)
validate_stringlist = _listify_validator(
    validate_string, doc='return a list of strings')
validate_int = _make_type_validator(int)
validate_int_or_None = _make_type_validator(int, allow_none=True)
validate_float = _make_type_validator(float)
validate_float_or_None = _make_type_validator(float, allow_none=True)
validate_floatlist = _listify_validator(
    validate_float, doc='return a list of floats')
_validate_standard_backends = ValidateInStrings(
    'backend', all_backends, ignorecase=True)
_auto_backend_sentinel = object()
validate_colorlist = _listify_validator(
    validate_color, allow_stringlist=True, doc='return a list of colorspecs')
validate_fontsizelist = _listify_validator(validate_fontsize)
_validate_named_linestyle = ValidateInStrings(
    'linestyle',
    [*ls_mapper.keys(), *ls_mapper.values(), 'None', 'none', ' ', ''],
    ignorecase=True)
validate_fillstyle = ValidateInStrings(
    'markers.fillstyle', ['full', 'left', 'right', 'bottom', 'top', 'none'])
validate_fillstylelist = _listify_validator(validate_fillstyle)
validate_markeverylist = _listify_validator(validate_markevery)
validate_hatchlist = _listify_validator(validate_hatch)
validate_dashlist = _listify_validator(validate_floatlist)
_prop_validators = {
        'color': _listify_validator(validate_color_for_prop_cycle,
                                    allow_stringlist=True),
        'linewidth': validate_floatlist,
        'linestyle': _listify_validator(_validate_linestyle),
        'facecolor': validate_colorlist,
        'edgecolor': validate_colorlist,
        'joinstyle': _listify_validator(JoinStyle),
        'capstyle': _listify_validator(CapStyle),
        'fillstyle': validate_fillstylelist,
        'markerfacecolor': validate_colorlist,
        'markersize': validate_floatlist,
        'markeredgewidth': validate_floatlist,
        'markeredgecolor': validate_colorlist,
        'markevery': validate_markeverylist,
        'alpha': validate_floatlist,
        'marker': validate_stringlist,
        'hatch': validate_hatchlist,
        'dashes': validate_dashlist,
    }
_prop_aliases = {
        'c': 'color',
        'lw': 'linewidth',
        'ls': 'linestyle',
        'fc': 'facecolor',
        'ec': 'edgecolor',
        'mfc': 'markerfacecolor',
        'mec': 'markeredgecolor',
        'mew': 'markeredgewidth',
        'ms': 'markersize',
    }
_validate_named_legend_loc = ValidateInStrings(
    'legend.loc',
    [
        "best",
        "upper right", "upper left", "lower left", "lower right", "right",
        "center left", "center right", "lower center", "upper center",
        "center"],
    ignorecase=True)
_validators = {
    "backend":           validate_backend,
    "backend_fallback":  validate_bool,
    "figure.hooks":      validate_stringlist,
    "toolbar":           _validate_toolbar,
    "interactive":       validate_bool,
    "timezone":          validate_string,

    "webagg.port":            validate_int,
    "webagg.address":         validate_string,
    "webagg.open_in_browser": validate_bool,
    "webagg.port_retries":    validate_int,

    # line props
    "lines.linewidth":       validate_float,  # line width in points
    "lines.linestyle":       _validate_linestyle,  # solid line
    "lines.color":           validate_color,  # first color in color cycle
    "lines.marker":          validate_string,  # marker name
    "lines.markerfacecolor": validate_color_or_auto,  # default color
    "lines.markeredgecolor": validate_color_or_auto,  # default color
    "lines.markeredgewidth": validate_float,
    "lines.markersize":      validate_float,  # markersize, in points
    "lines.antialiased":     validate_bool,  # antialiased (no jaggies)
    "lines.dash_joinstyle":  JoinStyle,
    "lines.solid_joinstyle": JoinStyle,
    "lines.dash_capstyle":   CapStyle,
    "lines.solid_capstyle":  CapStyle,
    "lines.dashed_pattern":  validate_floatlist,
    "lines.dashdot_pattern": validate_floatlist,
    "lines.dotted_pattern":  validate_floatlist,
    "lines.scale_dashes":    validate_bool,

    # marker props
    "markers.fillstyle": validate_fillstyle,

    ## pcolor(mesh) props:
    "pcolor.shading": ["auto", "flat", "nearest", "gouraud"],
    "pcolormesh.snap": validate_bool,

    ## patch props
    "patch.linewidth":       validate_float,  # line width in points
    "patch.edgecolor":       validate_color,
    "patch.force_edgecolor": validate_bool,
    "patch.facecolor":       validate_color,  # first color in cycle
    "patch.antialiased":     validate_bool,  # antialiased (no jaggies)

    ## hatch props
    "hatch.color":     validate_color,
    "hatch.linewidth": validate_float,

    ## Histogram properties
    "hist.bins": validate_hist_bins,

    ## Boxplot properties
    "boxplot.notch":       validate_bool,
    "boxplot.vertical":    validate_bool,
    "boxplot.whiskers":    validate_whiskers,
    "boxplot.bootstrap":   validate_int_or_None,
    "boxplot.patchartist": validate_bool,
    "boxplot.showmeans":   validate_bool,
    "boxplot.showcaps":    validate_bool,
    "boxplot.showbox":     validate_bool,
    "boxplot.showfliers":  validate_bool,
    "boxplot.meanline":    validate_bool,

    "boxplot.flierprops.color":           validate_color,
    "boxplot.flierprops.marker":          validate_string,
    "boxplot.flierprops.markerfacecolor": validate_color_or_auto,
    "boxplot.flierprops.markeredgecolor": validate_color,
    "boxplot.flierprops.markeredgewidth": validate_float,
    "boxplot.flierprops.markersize":      validate_float,
    "boxplot.flierprops.linestyle":       _validate_linestyle,
    "boxplot.flierprops.linewidth":       validate_float,

    "boxplot.boxprops.color":     validate_color,
    "boxplot.boxprops.linewidth": validate_float,
    "boxplot.boxprops.linestyle": _validate_linestyle,

    "boxplot.whiskerprops.color":     validate_color,
    "boxplot.whiskerprops.linewidth": validate_float,
    "boxplot.whiskerprops.linestyle": _validate_linestyle,

    "boxplot.capprops.color":     validate_color,
    "boxplot.capprops.linewidth": validate_float,
    "boxplot.capprops.linestyle": _validate_linestyle,

    "boxplot.medianprops.color":     validate_color,
    "boxplot.medianprops.linewidth": validate_float,
    "boxplot.medianprops.linestyle": _validate_linestyle,

    "boxplot.meanprops.color":           validate_color,
    "boxplot.meanprops.marker":          validate_string,
    "boxplot.meanprops.markerfacecolor": validate_color,
    "boxplot.meanprops.markeredgecolor": validate_color,
    "boxplot.meanprops.markersize":      validate_float,
    "boxplot.meanprops.linestyle":       _validate_linestyle,
    "boxplot.meanprops.linewidth":       validate_float,

    ## font props
    "font.family":     validate_stringlist,  # used by text object
    "font.style":      validate_string,
    "font.variant":    validate_string,
    "font.stretch":    validate_fontstretch,
    "font.weight":     validate_fontweight,
    "font.size":       validate_float,  # Base font size in points
    "font.serif":      validate_stringlist,
    "font.sans-serif": validate_stringlist,
    "font.cursive":    validate_stringlist,
    "font.fantasy":    validate_stringlist,
    "font.monospace":  validate_stringlist,

    # text props
    "text.color":          validate_color,
    "text.usetex":         validate_bool,
    "text.latex.preamble": validate_string,
    "text.hinting":        ["default", "no_autohint", "force_autohint",
                            "no_hinting", "auto", "native", "either", "none"],
    "text.hinting_factor": validate_int,
    "text.kerning_factor": validate_int,
    "text.antialiased":    validate_bool,
    "text.parse_math":     validate_bool,

    "mathtext.cal":            validate_font_properties,
    "mathtext.rm":             validate_font_properties,
    "mathtext.tt":             validate_font_properties,
    "mathtext.it":             validate_font_properties,
    "mathtext.bf":             validate_font_properties,
    "mathtext.bfit":           validate_font_properties,
    "mathtext.sf":             validate_font_properties,
    "mathtext.fontset":        ["dejavusans", "dejavuserif", "cm", "stix",
                                "stixsans", "custom"],
    "mathtext.default":        ["rm", "cal", "bfit", "it", "tt", "sf", "bf", "default",
                                "bb", "frak", "scr", "regular"],
    "mathtext.fallback":       _validate_mathtext_fallback,

    "image.aspect":          validate_aspect,  # equal, auto, a number
    "image.interpolation":   validate_string,
    "image.cmap":            _validate_cmap,  # gray, jet, etc.
    "image.lut":             validate_int,  # lookup table
    "image.origin":          ["upper", "lower"],
    "image.resample":        validate_bool,
    # Specify whether vector graphics backends will combine all images on a
    # set of axes into a single composite image
    "image.composite_image": validate_bool,

    # contour props
    "contour.negative_linestyle": _validate_linestyle,
    "contour.corner_mask":        validate_bool,
    "contour.linewidth":          validate_float_or_None,
    "contour.algorithm":          ["mpl2005", "mpl2014", "serial", "threaded"],

    # errorbar props
    "errorbar.capsize": validate_float,

    # axis props
    # alignment of x/y axis title
    "xaxis.labellocation": ["left", "center", "right"],
    "yaxis.labellocation": ["bottom", "center", "top"],

    # axes props
    "axes.axisbelow":        validate_axisbelow,
    "axes.facecolor":        validate_color,  # background color
    "axes.edgecolor":        validate_color,  # edge color
    "axes.linewidth":        validate_float,  # edge linewidth

    "axes.spines.left":      validate_bool,  # Set visibility of axes spines,
    "axes.spines.right":     validate_bool,  # i.e., the lines around the chart
    "axes.spines.bottom":    validate_bool,  # denoting data boundary.
    "axes.spines.top":       validate_bool,

    "axes.titlesize":     validate_fontsize,  # axes title fontsize
    "axes.titlelocation": ["left", "center", "right"],  # axes title alignment
    "axes.titleweight":   validate_fontweight,  # axes title font weight
    "axes.titlecolor":    validate_color_or_auto,  # axes title font color
    # title location, axes units, None means auto
    "axes.titley":        validate_float_or_None,
    # pad from axes top decoration to title in points
    "axes.titlepad":      validate_float,
    "axes.grid":          validate_bool,  # display grid or not
    "axes.grid.which":    ["minor", "both", "major"],  # which grids are drawn
    "axes.grid.axis":     ["x", "y", "both"],  # grid type
    "axes.labelsize":     validate_fontsize,  # fontsize of x & y labels
    "axes.labelpad":      validate_float,  # space between label and axis
    "axes.labelweight":   validate_fontweight,  # fontsize of x & y labels
    "axes.labelcolor":    validate_color,  # color of axis label
    # use scientific notation if log10 of the axis range is smaller than the
    # first or larger than the second
    "axes.formatter.limits": _listify_validator(validate_int, n=2),
    # use current locale to format ticks
    "axes.formatter.use_locale": validate_bool,
    "axes.formatter.use_mathtext": validate_bool,
    # minimum exponent to format in scientific notation
    "axes.formatter.min_exponent": validate_int,
    "axes.formatter.useoffset": validate_bool,
    "axes.formatter.offset_threshold": validate_int,
    "axes.unicode_minus": validate_bool,
    # This entry can be either a cycler object or a string repr of a
    # cycler-object, which gets eval()'ed to create the object.
    "axes.prop_cycle": validate_cycler,
    # If "data", axes limits are set close to the data.
    # If "round_numbers" axes limits are set to the nearest round numbers.
    "axes.autolimit_mode": ["data", "round_numbers"],
    "axes.xmargin": _validate_greaterthan_minushalf,  # margin added to xaxis
    "axes.ymargin": _validate_greaterthan_minushalf,  # margin added to yaxis
    "axes.zmargin": _validate_greaterthan_minushalf,  # margin added to zaxis

    "polaraxes.grid": validate_bool,  # display polar grid or not
    "axes3d.grid":    validate_bool,  # display 3d grid

    "axes3d.xaxis.panecolor":    validate_color,  # 3d background pane
    "axes3d.yaxis.panecolor":    validate_color,  # 3d background pane
    "axes3d.zaxis.panecolor":    validate_color,  # 3d background pane

    # scatter props
    "scatter.marker":     validate_string,
    "scatter.edgecolors": validate_string,

    "date.epoch": _validate_date,
    "date.autoformatter.year":        validate_string,
    "date.autoformatter.month":       validate_string,
    "date.autoformatter.day":         validate_string,
    "date.autoformatter.hour":        validate_string,
    "date.autoformatter.minute":      validate_string,
    "date.autoformatter.second":      validate_string,
    "date.autoformatter.microsecond": validate_string,

    'date.converter':          ['auto', 'concise'],
    # for auto date locator, choose interval_multiples
    'date.interval_multiples': validate_bool,

    # legend properties
    "legend.fancybox": validate_bool,
    "legend.loc": _validate_legend_loc,

    # the number of points in the legend line
    "legend.numpoints":      validate_int,
    # the number of points in the legend line for scatter
    "legend.scatterpoints":  validate_int,
    "legend.fontsize":       validate_fontsize,
    "legend.title_fontsize": validate_fontsize_None,
    # color of the legend
    "legend.labelcolor":     _validate_color_or_linecolor,
    # the relative size of legend markers vs. original
    "legend.markerscale":    validate_float,
    # using dict in rcParams not yet supported, so make sure it is bool
    "legend.shadow":         validate_bool,
    # whether or not to draw a frame around legend
    "legend.frameon":        validate_bool,
    # alpha value of the legend frame
    "legend.framealpha":     validate_float_or_None,

    ## the following dimensions are in fraction of the font size
    "legend.borderpad":      validate_float,  # units are fontsize
    # the vertical space between the legend entries
    "legend.labelspacing":   validate_float,
    # the length of the legend lines
    "legend.handlelength":   validate_float,
    # the length of the legend lines
    "legend.handleheight":   validate_float,
    # the space between the legend line and legend text
    "legend.handletextpad":  validate_float,
    # the border between the axes and legend edge
    "legend.borderaxespad":  validate_float,
    # the border between the axes and legend edge
    "legend.columnspacing":  validate_float,
    "legend.facecolor":      validate_color_or_inherit,
    "legend.edgecolor":      validate_color_or_inherit,

    # tick properties
    "xtick.top":           validate_bool,      # draw ticks on top side
    "xtick.bottom":        validate_bool,      # draw ticks on bottom side
    "xtick.labeltop":      validate_bool,      # draw label on top
    "xtick.labelbottom":   validate_bool,      # draw label on bottom
    "xtick.major.size":    validate_float,     # major xtick size in points
    "xtick.minor.size":    validate_float,     # minor xtick size in points
    "xtick.major.width":   validate_float,     # major xtick width in points
    "xtick.minor.width":   validate_float,     # minor xtick width in points
    "xtick.major.pad":     validate_float,     # distance to label in points
    "xtick.minor.pad":     validate_float,     # distance to label in points
    "xtick.color":         validate_color,     # color of xticks
    "xtick.labelcolor":    validate_color_or_inherit,  # color of xtick labels
    "xtick.minor.visible": validate_bool,      # visibility of minor xticks
    "xtick.minor.top":     validate_bool,      # draw top minor xticks
    "xtick.minor.bottom":  validate_bool,      # draw bottom minor xticks
    "xtick.major.top":     validate_bool,      # draw top major xticks
    "xtick.major.bottom":  validate_bool,      # draw bottom major xticks
    # number of minor xticks
    "xtick.minor.ndivs":   _validate_minor_tick_ndivs,
    "xtick.labelsize":     validate_fontsize,  # fontsize of xtick labels
    "xtick.direction":     ["out", "in", "inout"],  # direction of xticks
    "xtick.alignment":     ["center", "right", "left"],

    "ytick.left":          validate_bool,      # draw ticks on left side
    "ytick.right":         validate_bool,      # draw ticks on right side
    "ytick.labelleft":     validate_bool,      # draw tick labels on left side
    "ytick.labelright":    validate_bool,      # draw tick labels on right side
    "ytick.major.size":    validate_float,     # major ytick size in points
    "ytick.minor.size":    validate_float,     # minor ytick size in points
    "ytick.major.width":   validate_float,     # major ytick width in points
    "ytick.minor.width":   validate_float,     # minor ytick width in points
    "ytick.major.pad":     validate_float,     # distance to label in points
    "ytick.minor.pad":     validate_float,     # distance to label in points
    "ytick.color":         validate_color,     # color of yticks
    "ytick.labelcolor":    validate_color_or_inherit,  # color of ytick labels
    "ytick.minor.visible": validate_bool,      # visibility of minor yticks
    "ytick.minor.left":    validate_bool,      # draw left minor yticks
    "ytick.minor.right":   validate_bool,      # draw right minor yticks
    "ytick.major.left":    validate_bool,      # draw left major yticks
    "ytick.major.right":   validate_bool,      # draw right major yticks
    # number of minor yticks
    "ytick.minor.ndivs":   _validate_minor_tick_ndivs,
    "ytick.labelsize":     validate_fontsize,  # fontsize of ytick labels
    "ytick.direction":     ["out", "in", "inout"],  # direction of yticks
    "ytick.alignment":     [
        "center", "top", "bottom", "baseline", "center_baseline"],

    "grid.color":        validate_color,  # grid color
    "grid.linestyle":    _validate_linestyle,  # solid
    "grid.linewidth":    validate_float,     # in points
    "grid.alpha":        validate_float,

    ## figure props
    # figure title
    "figure.titlesize":   validate_fontsize,
    "figure.titleweight": validate_fontweight,

    # figure labels
    "figure.labelsize":   validate_fontsize,
    "figure.labelweight": validate_fontweight,

    # figure size in inches: width by height
    "figure.figsize":          _listify_validator(validate_float, n=2),
    "figure.dpi":              validate_float,
    "figure.facecolor":        validate_color,
    "figure.edgecolor":        validate_color,
    "figure.frameon":          validate_bool,
    "figure.autolayout":       validate_bool,
    "figure.max_open_warning": validate_int,
    "figure.raise_window":     validate_bool,
    "macosx.window_mode":      ["system", "tab", "window"],

    "figure.subplot.left":   validate_float,
    "figure.subplot.right":  validate_float,
    "figure.subplot.bottom": validate_float,
    "figure.subplot.top":    validate_float,
    "figure.subplot.wspace": validate_float,
    "figure.subplot.hspace": validate_float,

    "figure.constrained_layout.use": validate_bool,  # run constrained_layout?
    # wspace and hspace are fraction of adjacent subplots to use for space.
    # Much smaller than above because we don't need room for the text.
    "figure.constrained_layout.hspace": validate_float,
    "figure.constrained_layout.wspace": validate_float,
    # buffer around the axes, in inches.
    "figure.constrained_layout.h_pad": validate_float,
    "figure.constrained_layout.w_pad": validate_float,

    ## Saving figure's properties
    'savefig.dpi':          validate_dpi,
    'savefig.facecolor':    validate_color_or_auto,
    'savefig.edgecolor':    validate_color_or_auto,
    'savefig.orientation':  ['landscape', 'portrait'],
    "savefig.format":       validate_string,
    "savefig.bbox":         validate_bbox,  # "tight", or "standard" (= None)
    "savefig.pad_inches":   validate_float,
    # default directory in savefig dialog box
    "savefig.directory":    _validate_pathlike,
    "savefig.transparent":  validate_bool,

    "tk.window_focus": validate_bool,  # Maintain shell focus for TkAgg

    # Set the papersize/type
    "ps.papersize":       _validate_papersize,
    "ps.useafm":          validate_bool,
    # use ghostscript or xpdf to distill ps output
    "ps.usedistiller":    validate_ps_distiller,
    "ps.distiller.res":   validate_int,  # dpi
    "ps.fonttype":        validate_fonttype,  # 3 (Type3) or 42 (Truetype)
    "pdf.compression":    validate_int,  # 0-9 compression level; 0 to disable
    "pdf.inheritcolor":   validate_bool,  # skip color setting commands
    # use only the 14 PDF core fonts embedded in every PDF viewing application
    "pdf.use14corefonts": validate_bool,
    "pdf.fonttype":       validate_fonttype,  # 3 (Type3) or 42 (Truetype)

    "pgf.texsystem": ["xelatex", "lualatex", "pdflatex"],  # latex variant used
    "pgf.rcfonts":   validate_bool,  # use mpl's rc settings for font config
    "pgf.preamble":  validate_string,  # custom LaTeX preamble

    # write raster image data into the svg file
    "svg.image_inline": validate_bool,
    "svg.fonttype": ["none", "path"],  # save text as text ("none") or "paths"
    "svg.hashsalt": validate_string_or_None,

    # set this when you want to generate hardcopy docstring
    "docstring.hardcopy": validate_bool,

    "path.simplify":           validate_bool,
    "path.simplify_threshold": _validate_greaterequal0_lessequal1,
    "path.snap":               validate_bool,
    "path.sketch":             validate_sketch,
    "path.effects":            validate_anylist,
    "agg.path.chunksize":      validate_int,  # 0 to disable chunking

    # key-mappings (multi-character mappings should be a list/tuple)
    "keymap.fullscreen": validate_stringlist,
    "keymap.home":       validate_stringlist,
    "keymap.back":       validate_stringlist,
    "keymap.forward":    validate_stringlist,
    "keymap.pan":        validate_stringlist,
    "keymap.zoom":       validate_stringlist,
    "keymap.save":       validate_stringlist,
    "keymap.quit":       validate_stringlist,
    "keymap.quit_all":   validate_stringlist,  # e.g.: "W", "cmd+W", "Q"
    "keymap.grid":       validate_stringlist,
    "keymap.grid_minor": validate_stringlist,
    "keymap.yscale":     validate_stringlist,
    "keymap.xscale":     validate_stringlist,
    "keymap.help":       validate_stringlist,
    "keymap.copy":       validate_stringlist,

    # Animation settings
    "animation.html":         ["html5", "jshtml", "none"],
    # Limit, in MB, of size of base64 encoded animation in HTML
    # (i.e. IPython notebook)
    "animation.embed_limit":  validate_float,
    "animation.writer":       validate_string,
    "animation.codec":        validate_string,
    "animation.bitrate":      validate_int,
    # Controls image format when frames are written to disk
    "animation.frame_format": ["png", "jpeg", "tiff", "raw", "rgba", "ppm",
                               "sgi", "bmp", "pbm", "svg"],
    # Path to ffmpeg binary. If just binary name, subprocess uses $PATH.
    "animation.ffmpeg_path":  _validate_pathlike,
    # Additional arguments for ffmpeg movie writer (using pipes)
    "animation.ffmpeg_args":  validate_stringlist,
     # Path to convert binary. If just binary name, subprocess uses $PATH.
    "animation.convert_path": _validate_pathlike,
     # Additional arguments for convert movie writer (using pipes)
    "animation.convert_args": validate_stringlist,

    # Classic (pre 2.0) compatibility mode
    # This is used for things that are hard to make backward compatible
    # with a sane rcParam alone.  This does *not* turn on classic mode
    # altogether.  For that use `matplotlib.style.use("classic")`.
    "_internal.classic_mode": validate_bool
}
_hardcoded_defaults = {  # Defaults not inferred from
    # lib/matplotlib/mpl-data/matplotlibrc...
    # ... because they are private:
    "_internal.classic_mode": False,
    # ... because they are deprecated:
    # No current deprecations.
    # backend is handled separately when constructing rcParamsDefault.
}
_validators = {k: _convert_validator_spec(k, conv)
               for k, conv in _validators.items()}

def _validate_papersize(s):
    # Re-inline this validator when the 'auto' deprecation expires.
    s = ValidateInStrings("ps.papersize",
                          ["auto", "letter", "legal", "ledger",
                           *[f"{ab}{i}" for ab in "ab" for i in range(11)]],
                          ignorecase=True)(s)
    if s == "auto":
        _api.warn_deprecated("3.8", name="ps.papersize='auto'",
                             addendum="Pass an explicit paper type, or omit the "
                             "*ps.papersize* rcParam entirely.")
    return s
