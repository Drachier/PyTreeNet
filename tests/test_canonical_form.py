import unittest
import numpy as np

from copy import deepcopy

import pytreenet as ptn

from pytreenet.core.canonical_form import _find_smallest_distance_neighbour
from pytreenet.util.tensor_util import compute_transfer_tensor
from pytreenet.random import (random_small_ttns,
                              random_big_ttns_two_root_children)

class TestCanonicalFormSimple(unittest.TestCase):
    def setUp(self):
        self.ttn = random_small_ttns()

    def test_find_smallest_distance_neighbour_to_c1(self):
        distance_dict = self.ttn.distance_to_node("c1")
        root_node = self.ttn.root[0]
        mini_dist_neighbour_id = _find_smallest_distance_neighbour(root_node,
                                                                   distance_dict)
        self.assertEqual("c1", mini_dist_neighbour_id)
        node_c2 = self.ttn.nodes["c2"]
        mini_dist_neighbour_id = _find_smallest_distance_neighbour(node_c2,
                                                                   distance_dict)
        self.assertEqual("root", mini_dist_neighbour_id)

    def test_find_smallest_distance_neighbour_to_root(self):
        distance_dict = self.ttn.distance_to_node("root")
        node_c1 = self.ttn.nodes["c1"]
        mini_dist_neighbour_id = _find_smallest_distance_neighbour(node_c1,
                                                                   distance_dict)
        self.assertEqual("root", mini_dist_neighbour_id)
        node_c2 = self.ttn.nodes["c2"]
        mini_dist_neighbour_id = _find_smallest_distance_neighbour(node_c2,
                                                                   distance_dict)
        self.assertEqual("root", mini_dist_neighbour_id)

    def test_find_smallest_distance_neighbour_to_c2(self):
        distance_dict = self.ttn.distance_to_node("c2")
        root_node = self.ttn.root[0]
        mini_dist_neighbour_id = _find_smallest_distance_neighbour(root_node,
                                                                   distance_dict)
        self.assertEqual("c2", mini_dist_neighbour_id)
        node_c1 = self.ttn.nodes["c1"]
        mini_dist_neighbour_id = _find_smallest_distance_neighbour(node_c1,
                                                                   distance_dict)
        self.assertEqual("root", mini_dist_neighbour_id)

    def test_canoncial_form_c1_center(self):
        reference_ttn = deepcopy(self.ttn)

        ptn.canonical_form(self.ttn, "c1")

        ref_tensor = reference_ttn.completely_contract_tree()[0]
        found_tensor = self.ttn.completely_contract_tree(to_copy=True)[0]
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

        # Check, if root is isometry
        root_node, root_tensor = self.ttn.root
        contr_indices = [root_node.child_index("c2")]
        contr_indices.extend(root_node.open_legs)
        contr_indices = tuple(contr_indices)
        found_transfer_tensor = compute_transfer_tensor(root_tensor,
                                                            contr_indices)
        identity = np.eye(5)
        self.assertTrue(np.allclose(identity,found_transfer_tensor))

        # Check, if c2 is isometry
        c2_node, c2_tensor = self.ttn["c2"]
        contr_indices = tuple(c2_node.open_legs)
        found_transfer_tensor = compute_transfer_tensor(c2_tensor,
                                                            contr_indices)
        identity = np.eye(4)
        self.assertTrue(np.allclose(identity,found_transfer_tensor))

    def test_canoncial_form_root_center(self):
        reference_ttn = deepcopy(self.ttn)

        ptn.canonical_form(self.ttn, "root")

        ref_tensor = reference_ttn.completely_contract_tree()[0]
        found_tensor = self.ttn.completely_contract_tree(to_copy=True)[0]
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

        # Check, if c1 is isometry
        c1_node, c1_tensor = self.ttn["c1"]
        contr_indices = tuple(c1_node.open_legs)
        found_transfer_tensor = compute_transfer_tensor(c1_tensor,
                                                            contr_indices)
        identity = np.eye(3)
        self.assertTrue(np.allclose(identity,found_transfer_tensor))

        # Check, if c2 is isometry
        c2_node, c2_tensor = self.ttn["c2"]
        contr_indices = tuple(c2_node.open_legs)
        found_transfer_tensor = compute_transfer_tensor(c2_tensor,
                                                            contr_indices)
        identity = np.eye(4)
        self.assertTrue(np.allclose(identity,found_transfer_tensor))

    def test_canoncial_form_c2_center(self):
        reference_ttn = deepcopy(self.ttn)

        ptn.canonical_form(self.ttn, "c2")

        ref_tensor = reference_ttn.completely_contract_tree()[0]
        ref_tensor = np.transpose(ref_tensor, axes=(0,2,1))
        found_tensor = self.ttn.completely_contract_tree(to_copy=True)[0]
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

        # Check, if root is isometry
        root_node, root_tensor = self.ttn.root
        contr_indices = [root_node.child_index("c1")]
        contr_indices.extend(root_node.open_legs)
        contr_indices = tuple(contr_indices)
        found_transfer_tensor = compute_transfer_tensor(root_tensor,
                                                            contr_indices)
        identity = np.eye(6)
        self.assertTrue(np.allclose(identity,found_transfer_tensor))

        # Check, if c1 is isometry
        c1_node, c1_tensor = self.ttn["c1"]
        contr_indices = tuple(c1_node.open_legs)
        found_transfer_tensor = compute_transfer_tensor(c1_tensor,
                                                            contr_indices)
        identity = np.eye(3)
        self.assertTrue(np.allclose(identity,found_transfer_tensor))

    def test_canoncial_form_c1_center_keep(self):
        reference_ttn = deepcopy(self.ttn)
        ptn.canonical_form(self.ttn, "c1",
                           mode=ptn.SplitMode.KEEP)

        ref_tensor = reference_ttn.completely_contract_tree(to_copy=True)[0]
        found_tensor = self.ttn.completely_contract_tree(to_copy=True)[0]
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

        # Check, if root is isometry
        root_node, root_tensor = self.ttn.root
        contr_indices = [root_node.child_index("c2")]
        contr_indices.extend(root_node.open_legs)
        contr_indices = tuple(contr_indices)
        found_transfer_tensor = compute_transfer_tensor(root_tensor,
                                                            contr_indices)
        identity = np.eye(5)
        self.assertTrue(np.allclose(identity,found_transfer_tensor))
        # Check, if shape is kept
        correct_shape = reference_ttn.nodes["root"].shape
        found_shape = self.ttn.nodes["root"].shape
        self.assertEqual(correct_shape, found_shape)

        # Check, if c2 is isometry
        c2_node, c2_tensor = self.ttn["c2"]
        contr_indices = tuple(c2_node.open_legs)
        found_transfer_tensor = compute_transfer_tensor(c2_tensor,
                                                            contr_indices)
        identity = np.eye(4)
        identity = np.pad(identity, (0,2))
        self.assertTrue(np.allclose(identity,found_transfer_tensor))
        # Check, if shape is kept
        correct_shape = reference_ttn.nodes["c2"].shape
        found_shape = self.ttn.nodes["c2"].shape
        self.assertEqual(correct_shape, found_shape)

    def test_canoncial_form_root_center_keep(self):
        reference_ttn = deepcopy(self.ttn)
        ptn.canonical_form(self.ttn, "root",
                           mode=ptn.SplitMode.KEEP)

        ref_tensor = reference_ttn.completely_contract_tree(to_copy=True)[0]
        found_tensor = self.ttn.completely_contract_tree(to_copy=True)[0]
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

        # Check, if c1 is isometry
        c1_node, c1_tensor = self.ttn["c1"]
        contr_indices = tuple(c1_node.open_legs)
        found_transfer_tensor = compute_transfer_tensor(c1_tensor,
                                                            contr_indices)
        identity = np.eye(3)
        identity = np.pad(identity, (0,2))
        self.assertTrue(np.allclose(identity,found_transfer_tensor))
        # Check, if shape is kept
        correct_shape = reference_ttn.nodes["c1"].shape
        found_shape = self.ttn.nodes["c1"].shape
        self.assertEqual(correct_shape, found_shape)

        # Check, if c2 is isometry
        c2_node, c2_tensor = self.ttn["c2"]
        contr_indices = tuple(c2_node.open_legs)
        found_transfer_tensor = compute_transfer_tensor(c2_tensor,
                                                            contr_indices)
        identity = np.eye(4)
        identity = np.pad(identity, (0,2))
        self.assertTrue(np.allclose(identity,found_transfer_tensor))
        # Check, if shape is kept
        correct_shape = reference_ttn.nodes["c2"].shape
        found_shape = self.ttn.nodes["c2"].shape
        self.assertEqual(correct_shape, found_shape)

    def test_canoncial_form_c2_center_keep(self):
        reference_ttn = deepcopy(self.ttn)
        ptn.canonical_form(self.ttn, "c2",
                           mode=ptn.SplitMode.KEEP)

        ref_tensor = reference_ttn.completely_contract_tree(to_copy=True)[0]
        ref_tensor = np.transpose(ref_tensor, axes=(0,2,1))
        found_tensor = self.ttn.completely_contract_tree(to_copy=True)[0]
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

        # Check, if root is isometry
        root_node, root_tensor = self.ttn.root
        contr_indices = [root_node.child_index("c1")]
        contr_indices.extend(root_node.open_legs)
        contr_indices = tuple(contr_indices)
        found_transfer_tensor = compute_transfer_tensor(root_tensor,
                                                            contr_indices)
        identity = np.eye(6)
        self.assertTrue(np.allclose(identity,found_transfer_tensor))
        # Check, if shape is kept
        correct_shape = reference_ttn.nodes["root"].shape
        correct_shape = tuple([correct_shape[i] for i in (1,0,2)])
        found_shape = self.ttn.nodes["root"].shape
        self.assertEqual(correct_shape, found_shape)

        # Check, if c1 is isometry
        c1_node, c1_tensor = self.ttn["c1"]
        contr_indices = tuple(c1_node.open_legs)
        found_transfer_tensor = compute_transfer_tensor(c1_tensor,
                                                            contr_indices)
        identity = np.eye(3)
        identity = np.pad(identity, (0,2))
        self.assertTrue(np.allclose(identity,found_transfer_tensor))
        # Check, if shape is kept
        correct_shape = reference_ttn.nodes["c1"].shape
        found_shape = self.ttn.nodes["c1"].shape
        self.assertEqual(correct_shape, found_shape)

class TestCanoncialFormComplicated(unittest.TestCase):
    def setUp(self) -> None:
        self.ttn = random_big_ttns_two_root_children()
        self.ref_ttn = deepcopy(self.ttn)

    def test_canoncial_form_0_center(self):
        center_id = "site" + str(0)
        ptn.canonical_form(self.ttn, center_id)

        ref_tensor = self.ref_ttn.completely_contract_tree()[0]
        found_tensor = self.ttn.completely_contract_tree(to_copy=True)[0]
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_canoncial_form_1_center(self):
        center_id = "site" + str(1)
        ptn.canonical_form(self.ttn, center_id)

        ref_tensor = self.ref_ttn.completely_contract_tree()[0]
        found_tensor = self.ttn.completely_contract_tree(to_copy=True)[0]
        self.assertTrue(np.allclose(ref_tensor,found_tensor))


if __name__ == "__main__":
    unittest.main()
