from __future__ import print_function, division
from sympy import floor, pi, sqrt, sympify, eye
from sympy.core.compatibility import range
from sympy.core.numbers import NegativeOne
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import UnitaryOperator
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import IntQubit

__all__ = [
    'OracleGate',
    'WGate',
    'superposition_basis',
    'grover_iteration',
    'apply_grover'
]

class OracleGate(Gate):
    gate_name = u'V'
    gate_name_latex = u'V'
    @property
    def search_function(self):
        """The unknown function that helps find the sought after qubits."""
        return self.label[1]
    @property
    def targets(self):
        """A tuple of target qubits."""
        return sympify(tuple(range(self.args[0])))
    def _represent_ZGate(self, basis, **options):
        """
        Represent the OracleGate in the computational basis.
        """
        nbasis = 2**self.nqubits  # compute it only once
        matrixOracle = eye(nbasis)
        # Flip the sign given the output of the oracle function
        for i in range(nbasis):
            if self.search_function(IntQubit(i, nqubits=self.nqubits)):
                matrixOracle[i, i] = NegativeOne()
        return matrixOracle