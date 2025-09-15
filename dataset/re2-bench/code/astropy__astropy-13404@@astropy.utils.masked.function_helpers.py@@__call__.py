import numpy as np
from astropy.units.quantity_helper.function_helpers import (
    FunctionAssigner)
from astropy.utils.compat import NUMPY_LT_1_19, NUMPY_LT_1_20, NUMPY_LT_1_23
from .core import Masked
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked
from .core import Masked
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked
from .core import Masked
from numpy.core.arrayprint import _leading_trailing, _formatArray
from numpy.core.arrayprint import _make_options_dict, _format_options
from numpy.core.arrayprint import _get_format_function
from astropy.utils.masked import Masked
from astropy.utils.masked import Masked

__all__ = ['MASKED_SAFE_FUNCTIONS', 'APPLY_TO_BOTH_FUNCTIONS',
           'DISPATCHED_FUNCTIONS', 'UNSUPPORTED_FUNCTIONS']
MASKED_SAFE_FUNCTIONS = set()
APPLY_TO_BOTH_FUNCTIONS = {}
DISPATCHED_FUNCTIONS = {}
UNSUPPORTED_FUNCTIONS = set()
IGNORED_FUNCTIONS = {
    # I/O - useless for Masked, since no way to store the mask.
    np.save, np.savez, np.savetxt, np.savez_compressed,
    # Polynomials
    np.poly, np.polyadd, np.polyder, np.polydiv, np.polyfit, np.polyint,
    np.polymul, np.polysub, np.polyval, np.roots, np.vander}
apply_to_both = FunctionAssigner(APPLY_TO_BOTH_FUNCTIONS)
dispatched_function = FunctionAssigner(DISPATCHED_FUNCTIONS)
_nanfunc_fill_values = {'nansum': 0, 'nancumsum': 0,
                        'nanprod': 1, 'nancumprod': 1}

class MaskedFormat:
    def __call__(self, x):
        if x.dtype.names:
            # The replacement of x with a list is needed because the function
            # inside StructuredVoidFormat iterates over x, which works for an
            # np.void but not an array scalar.
            return self.format_function([x[field] for field in x.dtype.names])

        if x.shape:
            # For a subarray pass on the data directly, since the
            # items will be iterated on inside the function.
            return self.format_function(x)

        # Single element: first just typeset it normally, replace with masked
        # string if needed.
        string = self.format_function(x.unmasked[()])
        if x.mask:
            # Strikethrough would be neat, but terminal needs a different
            # formatting than, say, jupyter notebook.
            # return "\x1B[9m"+string+"\x1B[29m"
            # return ''.join(s+'\u0336' for s in string)
            n = min(3, max(1, len(string)))
            return ' ' * (len(string)-n) + '\u2014' * n
        else:
            return string