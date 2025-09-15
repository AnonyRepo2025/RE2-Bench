import numpy as np
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import weakref
from astropy import log
from astropy.units import Unit, Quantity, UnitConversionError

__all__ = ['MissingDataAssociationException',
           'IncompatibleUncertaintiesException', 'NDUncertainty',
           'StdDevUncertainty', 'UnknownUncertainty',
           'VarianceUncertainty', 'InverseVariance']

class VarianceUncertainty(_VariancePropagationMixin, NDUncertainty):
    def _convert_to_variance(self):
        return self