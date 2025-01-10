from unittest import TestCase, main
from copy import deepcopy

from numpy import allclose

from pytreenet.ttns.ttndo import SymmetricTTNDO
from pytreenet.random.random_ttndo import (generate_one_child_layer_ttdndo,
                                           generate_three_child_layers_ttdndo)

from pytreenet.contractions.ttndo_contractions import (ttndo_contraction_order,
                                                       trace_ttndo)

class TestTTNDOContractionOrder(TestCase):
    """
    Test the function ttndo_contraction_order.
    """

    def test_for_empty_ttndo(self):
        """
        Test the contraction order for an empty TTNDO.
        """
        ttndo = SymmetricTTNDO()
        self.assertEqual(ttndo_contraction_order(ttndo), [])

    def test_for_root_only(self):
        """
        Test the contraction order for a TTNDO with only a root node.
        """
        ttndo = SymmetricTTNDO()
        ttndo.add_trivial_root("root")
        self.assertEqual(ttndo_contraction_order(ttndo), [])

    def test_for_one_child_layer(self):
        """
        Test the contraction order for a TTNDO with one bra and ket child each.
        """
        ttndo = generate_one_child_layer_ttdndo()
        self.assertEqual(ttndo_contraction_order(ttndo), ["child_ly1_ket"])

    def test_for_three_child_layers(self):
        """
        Test the contraction order for a TTNDO with three child layers.
        
        This also includes a node with three children and a node with a child
        and an open dimension.
        
        """
        ttndo = generate_three_child_layers_ttdndo()
        correct_order = ["child_ly2_ket",
                         "child_ly10_ket",
                         "child_ly11_ket",
                         "child_ly12_ket",
                         "child_ly0_ket"]
        self.assertEqual(ttndo_contraction_order(ttndo), correct_order)

class TestTraceTTNDO(TestCase):
    """
    Test the function trace_ttndo.
    """

    def test_for_emtpy_ttndo(self):
        """
        Test the trace of an empty TTNDO.
        """
        ttndo = SymmetricTTNDO()
        self.assertEqual(trace_ttndo(ttndo), 0)

    def test_for_root_only(self):
        """
        Test the trace of a TTNDO with only a root node.
        """
        ttndo = SymmetricTTNDO()
        ttndo.add_trivial_root("root")
        self.assertRaises(ValueError, trace_ttndo, ttndo)

    def test_for_one_child_layer(self):
        """
        Test the trace of a TTNDO with one bra and ket child each.
        """
        ttndo = generate_one_child_layer_ttdndo()
        # Compute Reference
        ref_ttndo = deepcopy(ttndo)
        density_marix, _ = ref_ttndo.completely_contract_tree()
        ref_trace = density_marix.reshape((3,3)).trace()
        # Test
        found_trace = trace_ttndo(ttndo)
        self.assertTrue(allclose(found_trace, ref_trace))

    def test_for_three_child_layers(self):
        """
        Test the trace of a TTNDO with three child layers.
        
        This also includes a node with three children and a node with a child
        and an open dimension.
        
        """
        ttndo = generate_three_child_layers_ttdndo()
        # Compute Reference
        ref_ttndo = deepcopy(ttndo)
        density_tensor, _ = ref_ttndo.completely_contract_tree()
        phys_dim = 2*2*2*5
        density_matrix = density_tensor.reshape((phys_dim, phys_dim))
        ref_trace = density_matrix.trace()
        # Test
        found_trace = trace_ttndo(ttndo)
        self.assertTrue(allclose(found_trace, ref_trace))

if __name__ == "__main__":
    main()
