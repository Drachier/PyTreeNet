from unittest import main as unitmain, TestCase

from numpy import zeros, allclose
import numpy.testing as npt

from pytreenet.special_ttn.binary import (generate_binary_ttns,
                                          optimised_2d_binary_ttn)
from pytreenet.operators.common_operators import ket_i
from pytreenet.special_ttn.special_nodes import constant_bd_trivial_node

class TestGenerateBinaryTTNS(TestCase):
    """
    Tests the generation of a binary TTNS under various aspects.
    """

    def test_trivial_generation(self):
        """
        Tests the generation of a binary TTNS with a single node.
        """
        ttns = generate_binary_ttns(1,
                                    2,
                                    ket_i(0,3)
                                    )
        self.assertEqual(1, len(ttns.nodes))
        self.assertEqual(1, len(ttns.tensors))
        self.assertTrue(allclose(ttns.tensors[ttns.root_id],
                                 ket_i(0,3)))

class TestOptimised2DBinaryTTN(TestCase):
    """
    Tests the generation of an optimised 2D binary TTN.
    """

    def test_2x2_system(self):
        """
        Tests the generation of an optimised 2D binary TTN for a 2x2 system.
        """
        ttns = optimised_2d_binary_ttn(2,
                                       2,
                                       ket_i(0,3)
                                       )
        self.assertEqual(7, len(ttns.nodes))
        self.assertEqual(7, len(ttns.tensors))
        # Check that it represents the correct state
        state, _ = ttns.completely_contract_tree(to_copy=True)
        npt.assert_allclose(state.flatten(),
                            ket_i(0,3**4))

    def test_3x3_system(self):
        """
        Tests the generation of an optimised 2D binary TTN for a 3x3 system.
        """
        ttns = optimised_2d_binary_ttn(3,
                                       2,
                                       ket_i(0,3)
                                       )
        self.assertEqual(17, len(ttns.nodes))
        self.assertEqual(17, len(ttns.tensors))
        # Check that it represents the correct state
        state, _ = ttns.completely_contract_tree(to_copy=True)
        npt.assert_allclose(state.flatten(),
                            ket_i(0,3**9))

    def test_4x4_system(self):
        """
        Tests the generation of an optimised 2D binary TTN for a 4x4 system.
        """
        ttns = optimised_2d_binary_ttn(4,
                                       2,
                                       ket_i(0,2)
                                       )
        self.assertEqual(31, len(ttns.nodes))
        self.assertEqual(31, len(ttns.tensors))
        # Check that it represents the correct state
        state, _ = ttns.completely_contract_tree(to_copy=True)
        npt.assert_allclose(state.flatten(),
                            ket_i(0,2**16))

    def test_5x5_system(self):
        """
        Tests the generation of an optimised 2D binary TTN for a 5x5 system.
        """
        ttns = optimised_2d_binary_ttn(5,
                                       2,
                                       ket_i(0,2)
                                       )
        self.assertEqual(49, len(ttns.nodes))
        self.assertEqual(49, len(ttns.tensors))
        # Check that it represents the correct state
        state, _ = ttns.completely_contract_tree(to_copy=True)
        npt.assert_allclose(state.flatten(),
                            ket_i(0,2**25))

if __name__ == "__main__":
    unitmain()
