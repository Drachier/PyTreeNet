import unittest
import numpy as np

import pytreenet as ptn


class TestTreeTensorNetwork(unittest.TestCase):
    def test_swaps(self):

        ref_swap = np.asarray([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]], dtype=complex)
        found_swap = ptn.swap_gate()

        self.assertTrue(np.allclose(ref_swap, found_swap))

if __name__ == "__main__":
    unittest.main()