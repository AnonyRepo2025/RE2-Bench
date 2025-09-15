import array
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
import numbers
from operator import itemgetter
import re
import unicodedata
import warnings
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, TransformerMixin
from ..preprocessing import normalize
from .hashing import FeatureHasher
from .stop_words import ENGLISH_STOP_WORDS
from ..utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
from ..utils import _IS_32BIT
from ..utils.fixes import _astype_copy_false
from ..exceptions import ChangedBehaviorWarning
import tempfile

__all__ = ['HashingVectorizer',
           'CountVectorizer',
           'ENGLISH_STOP_WORDS',
           'TfidfTransformer',
           'TfidfVectorizer',
           'strip_accents_ascii',
           'strip_accents_unicode',
           'strip_tags']

def _preprocess(doc, accent_function=None, lower=False):
    """Chain together an optional series of text preprocessing steps to
    apply to a document.

    Parameters
    ----------
    doc: str
        The string to preprocess
    accent_function: callable
        Function for handling accented characters. Common strategies include
        normalizing and removing.
    lower: bool
        Whether to use str.lower to lowercase all fo the text

    Returns
    -------
    doc: str
        preprocessed string
    """
    if lower:
        doc = doc.lower()
    if accent_function is not None:
        doc = accent_function(doc)
    return doc
