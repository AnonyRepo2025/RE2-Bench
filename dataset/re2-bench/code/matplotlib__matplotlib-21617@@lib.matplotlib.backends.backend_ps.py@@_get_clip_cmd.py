import codecs
import datetime
from enum import Enum
import functools
import glob
from io import StringIO, TextIOWrapper
import logging
import math
import os
import pathlib
import tempfile
import re
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib.afm import AFM
from matplotlib.backend_bases import (
    _Backend, _check_savefig_extra_args, FigureCanvasBase, FigureManagerBase,
    GraphicsContextBase, RendererBase)
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_HINTING, LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib.mathtext import MathTextParser
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps

_log = logging.getLogger(__name__)
backend_version = 'Level II'
debugPS = False
ps_backend_helper = PsBackendHelper()
papersize = {'letter': (8.5, 11),
             'legal': (8.5, 14),
             'ledger': (11, 17),
             'a0': (33.11, 46.81),
             'a1': (23.39, 33.11),
             'a2': (16.54, 23.39),
             'a3': (11.69, 16.54),
             'a4': (8.27, 11.69),
             'a5': (5.83, 8.27),
             'a6': (4.13, 5.83),
             'a7': (2.91, 4.13),
             'a8': (2.07, 2.91),
             'a9': (1.457, 2.05),
             'a10': (1.02, 1.457),
             'b0': (40.55, 57.32),
             'b1': (28.66, 40.55),
             'b2': (20.27, 28.66),
             'b3': (14.33, 20.27),
             'b4': (10.11, 14.33),
             'b5': (7.16, 10.11),
             'b6': (5.04, 7.16),
             'b7': (3.58, 5.04),
             'b8': (2.51, 3.58),
             'b9': (1.76, 2.51),
             'b10': (1.26, 1.76)}
FigureManagerPS = FigureManagerBase
psDefs = [
    # name proc  *_d*  -
    # Note that this cannot be bound to /d, because when embedding a Type3 font
    # we may want to define a "d" glyph using "/d{...} d" which would locally
    # overwrite the definition.
    "/_d { bind def } bind def",
    # x y  *m*  -
    "/m { moveto } _d",
    # x y  *l*  -
    "/l { lineto } _d",
    # x y  *r*  -
    "/r { rlineto } _d",
    # x1 y1 x2 y2 x y *c*  -
    "/c { curveto } _d",
    # *cl*  -
    "/cl { closepath } _d",
    # *ce*  -
    "/ce { closepath eofill } _d",
    # w h x y  *box*  -
    """/box {
      m
      1 index 0 r
      0 exch r
      neg 0 r
      cl
    } _d""",
    # w h x y  *clipbox*  -
    """/clipbox {
      box
      clip
      newpath
    } _d""",
    # wx wy llx lly urx ury  *setcachedevice*  -
    "/sc { setcachedevice } _d",
]

class RendererPS(RendererPDFPSBase):
    _afm_font_dir = cbook._get_data_path("fonts/afm")
    _use_afm_rc_name = "ps.useafm"
    mathtext_parser = _api.deprecated("3.4")(property(
        lambda self: MathTextParser("PS")))
    def _convert_path(self, path, transform, clip=False, simplify=None):
        if clip:
            clip = (0.0, 0.0, self.width * 72.0, self.height * 72.0)
        else:
            clip = None
        return _path.convert_to_string(
            path, transform, clip, simplify, None,
            6, [b"m", b"l", b"", b"c", b"cl"], True).decode("ascii")
    def _get_clip_cmd(self, gc):
        clip = []
        rect = gc.get_clip_rectangle()
        if rect is not None:
            clip.append("%s clipbox\n" % _nums_to_str(*rect.size, *rect.p0))
        path, trf = gc.get_clip_path()
        if path is not None:
            key = (path, id(trf))
            custom_clip_cmd = self._clip_paths.get(key)
            if custom_clip_cmd is None:
                custom_clip_cmd = "c%d" % len(self._clip_paths)
                self._pswriter.write(f"""\
/{custom_clip_cmd} {{
{self._convert_path(path, trf, simplify=False)}
clip
newpath
}} bind def
""")
                self._clip_paths[key] = custom_clip_cmd
            clip.append(f"{custom_clip_cmd}\n")
        return "".join(clip)