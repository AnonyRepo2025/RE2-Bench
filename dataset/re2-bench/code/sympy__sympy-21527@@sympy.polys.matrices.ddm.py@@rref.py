from .exceptions import DDMBadInputError, DDMShapeError, DDMDomainError
from .dense import (
        ddm_transpose,
        ddm_iadd,
        ddm_isub,
        ddm_ineg,
        ddm_imul,
        ddm_imatmul,
        ddm_irref,
        ddm_idet,
        ddm_iinv,
        ddm_ilu_split,
        ddm_ilu_solve,
        ddm_berk,
        )
from .sdm import SDM



class DDM(list):
    fmt = 'dense'
    def __init__(self, rowslist, shape, domain):
        super().__init__(rowslist)
        self.shape = self.rows, self.cols = m, n = shape
        self.domain = domain

        if not (len(self) == m and all(len(row) == n for row in self)):
            raise DDMBadInputError("Inconsistent row-list/shape")
    def copy(self):
        copyrows = (row[:] for row in self)
        return DDM(copyrows, self.shape, self.domain)
    def rref(a):
        """Reduced-row echelon form of a and list of pivots"""
        b = a.copy()
        K = a.domain
        partial_pivot = K.is_RealField or K.is_ComplexField
        pivots = ddm_irref(b, _partial_pivot=partial_pivot)
        return b, pivots