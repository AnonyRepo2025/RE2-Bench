import re
import warnings
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from . import conf
from .util import _is_int, _str_to_num, _words_group, translate
from .verify import VerifyError, VerifyWarning, _ErrList, _Verify

__all__ = ["Card", "Undefined"]
FIX_FP_TABLE = str.maketrans("de", "DE")
FIX_FP_TABLE2 = str.maketrans("dD", "eE")
CARD_LENGTH = 80
BLANK_CARD = " " * CARD_LENGTH
KEYWORD_LENGTH = 8  # The max length for FITS-standard keywords
VALUE_INDICATOR = "= "  # The standard FITS value indicator
VALUE_INDICATOR_LEN = len(VALUE_INDICATOR)
HIERARCH_VALUE_INDICATOR = "="  # HIERARCH cards may use a shortened indicator
UNDEFINED = Undefined()

class Card(_Verify):
    length = CARD_LENGTH
    _keywd_FSC_RE = re.compile(r"^[A-Z0-9_-]{0,%d}$" % KEYWORD_LENGTH)
    _keywd_hierarch_RE = re.compile(r"^(?:HIERARCH +)?(?:^[ -<>-~]+ ?)+$", re.I)
    _digits_FSC = r"(\.\d+|\d+(\.\d*)?)([DE][+-]?\d+)?"
    _digits_NFSC = r"(\.\d+|\d+(\.\d*)?) *([deDE] *[+-]? *\d+)?"
    _numr_FSC = r"[+-]?" + _digits_FSC
    _numr_NFSC = r"[+-]? *" + _digits_NFSC
    _number_FSC_RE = re.compile(rf"(?P<sign>[+-])?0*?(?P<digt>{_digits_FSC})")
    _number_NFSC_RE = re.compile(rf"(?P<sign>[+-])? *0*?(?P<digt>{_digits_NFSC})")
    _strg = r"\'(?P<strg>([ -~]+?|\'\'|) *?)\'(?=$|/| )"
    _comm_field = r"(?P<comm_field>(?P<sepr>/ *)(?P<comm>(.|\n)*))"
    _strg_comment_RE = re.compile(f"({_strg})? *{_comm_field}?$")
    _ascii_text_re = re.compile(r"[ -~]*\Z")
    _value_FSC_RE = re.compile(
        r'(?P<valu_field> *'
            r'(?P<valu>'

                #  The <strg> regex is not correct for all cases, but
                #  it comes pretty darn close.  It appears to find the
                #  end of a string rather well, but will accept
                #  strings with an odd number of single quotes,
                #  instead of issuing an error.  The FITS standard
                #  appears vague on this issue and only states that a
                #  string should not end with two single quotes,
                #  whereas it should not end with an even number of
                #  quotes to be precise.
                #
                #  Note that a non-greedy match is done for a string,
                #  since a greedy match will find a single-quote after
                #  the comment separator resulting in an incorrect
                #  match.
                rf'{_strg}|'
                r'(?P<bool>[FT])|'
                r'(?P<numr>' + _numr_FSC + r')|'
                r'(?P<cplx>\( *'
                    r'(?P<real>' + _numr_FSC + r') *, *'
                    r'(?P<imag>' + _numr_FSC + r') *\))'
            r')? *)'
        r'(?P<comm_field>'
            r'(?P<sepr>/ *)'
            r'(?P<comm>[!-~][ -~]*)?'
        r')?$'
    )
    _value_NFSC_RE = re.compile(
        r'(?P<valu_field> *'
            r'(?P<valu>'
                rf'{_strg}|'
                r'(?P<bool>[FT])|'
                r'(?P<numr>' + _numr_NFSC + r')|'
                r'(?P<cplx>\( *'
                    r'(?P<real>' + _numr_NFSC + r') *, *'
                    r'(?P<imag>' + _numr_NFSC + r') *\))'
            fr')? *){_comm_field}?$'
    )
    _rvkc_identifier = r"[a-zA-Z_]\w*"
    _rvkc_field = _rvkc_identifier + r"(\.\d+)?"
    _rvkc_field_specifier_s = rf"{_rvkc_field}(\.{_rvkc_field})*"
    _rvkc_field_specifier_val = r"(?P<keyword>{}): +(?P<val>{})".format(
        _rvkc_field_specifier_s, _numr_FSC
    )
    _rvkc_keyword_val = rf"\'(?P<rawval>{_rvkc_field_specifier_val})\'"
    _rvkc_keyword_val_comm = rf" *{_rvkc_keyword_val} *(/ *(?P<comm>[ -~]*))?$"
    _rvkc_field_specifier_val_RE = re.compile(_rvkc_field_specifier_val + "$")
    _rvkc_keyword_name_RE = re.compile(
        r"(?P<keyword>{})\.(?P<field_specifier>{})$".format(
            _rvkc_identifier, _rvkc_field_specifier_s
        )
    )
    _rvkc_keyword_val_comm_RE = re.compile(_rvkc_keyword_val_comm)
    _commentary_keywords = {"", "COMMENT", "HISTORY", "END"}
    _special_keywords = _commentary_keywords.union(["CONTINUE"])
    _value_indicator = VALUE_INDICATOR
    @value.deleter
    def value(self):
        if self._invalid:
            raise ValueError(
                "The value of invalid/unparsable cards cannot deleted.  "
                "Either delete this card from the header or replace it."
            )

        if not self.field_specifier:
            self.value = ""
        else:
            raise AttributeError(
                "Values cannot be deleted from record-valued keyword cards"
            )