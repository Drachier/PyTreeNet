import unittest
import numpy as np
from scipy.linalg import expm

import pytreenet as ptn

pauli = [np.array([[0, 1], [1, 0]], dtype=complex),
         np.array([[0, -1j], [1j, 0]], dtype=complex),
         np.array([[1, 0], [0, -1]], dtype=complex)]


def test_fast_exp_action(size):
    mat = np.array([1])
    vector = 1 / np.sqrt(2**size) * np.ones((2**size))
    for _ in range(size):
        operator_id = np.random.choice(3, 1)[0]
        mat = np.kron(mat, pauli[operator_id])
    for mode in ["expm", "eigsh", "chebyshev", "none"]:
        res = ptn.fast_exp_action(mat, vector, mode)
        assert res == expm(mat) @vector
    return 


class TestTreeTensorNetwork(unittest.TestCase):
    def test_all_modes(self):
        for size in [3, 4, 5, 6]:
            test_fast_exp_action(size)

if __name__ == "__main__":
    unittest.main()