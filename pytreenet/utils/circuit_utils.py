from .util import *

X, Y, Z = pauli_matrices()
one = one()
zero = zero()
I = np.eye(2, dtype=complex)
H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
O = 0*I
minus = H @ one
plus = H @ zero
if_zero = np.outer(zero, zero)
if_one = np.outer(one, one)