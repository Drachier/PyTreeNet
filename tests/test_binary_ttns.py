from unittest import main as unitmain, TestCase

from numpy import zeros, allclose

from pytreenet.special_ttn.binary import generate_binary_ttns
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

if __name__ == "__main__":
    unitmain()
