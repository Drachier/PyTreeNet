import unittest
import numpy as np

import pytreenet as ptn

from pytreenet.canonical_form import _find_smallest_distance_neighbour

class TestCanonicalFormSimple(unittest.TestCase):
    def setUp(self):
        self.tree_tensor_network = ptn.TreeTensorNetwork()

        self.node1 = ptn.random_tensor_node((2,3), identifier="node1")
        self.node2 = ptn.random_tensor_node((2,4,5), identifier="node2")

        self.tree_tensor_network.add_root(self.node1)
        self.tree_tensor_network.add_child_to_parent(self.node2, 0, "node1", 0)

    def testsimple_find_smallest_distance_neighbout(self):
        distance_dict = self.tree_tensor_network.distance_to_node("node1")
        minimum_distance_neighbour_id = _find_smallest_distance_neighbour(self.node2, distance_dict)
        self.assertEqual("node1", minimum_distance_neighbour_id)

    def testsimple_canonical_form(self):
        reference_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        ref_tensor = reference_ttn.nodes[reference_ttn.root_id].tensor
        
        node1 = self.tree_tensor_network.nodes["node1"]
        node2 = self.tree_tensor_network.nodes["node2"]
        ref_tensor_direct = np.tensordot(node1.tensor, node2.tensor, axes=([0],[0]))
        self.assertTrue(np.allclose(ref_tensor, ref_tensor_direct))

        ptn.canonical_form(self.tree_tensor_network, "node1")

        result_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        result_tensor = result_ttn.nodes[result_ttn.root_id].tensor

        self.assertFalse(result_ttn == reference_ttn)
        self.assertTrue(np.allclose(ref_tensor,result_tensor))

        node2 = self.tree_tensor_network.nodes["node2"]
        tensor2 = node2.tensor
        parent_leg2 = node2.parent_leg
        parent_dimension2 = tensor2.shape[parent_leg2[1]]
        identity = np.eye(parent_dimension2)

        open_indices2 = node2.open_legs
        transfer_tensor = ptn.compute_transfer_tensor(tensor2, open_indices2)

        self.assertEqual(transfer_tensor.shape, (2,2))
        self.assertTrue(np.allclose(identity, transfer_tensor))


class TestCanonicalFormComplicated(unittest.TestCase):
    def setUp(self):
        self.tree_tensor_network = ptn.TreeTensorNetwork()

        # Constructing a tree for tests
        self.node1 = ptn.random_tensor_node((2,2), identifier="node1")
        self.node2 = ptn.random_tensor_node((2,3,3), identifier="node2")
        self.node3 = ptn.random_tensor_node((3,), identifier="node3")
        self.node4 = ptn.random_tensor_node((3,4), identifier="node4")
        self.node5 = ptn.random_tensor_node((2,3,3), identifier="node5")
        self.node6 = ptn.random_tensor_node((3,4,4), identifier="node6")
        self.node7 = ptn.random_tensor_node((4,2,2), identifier="node7")

        self.tree_tensor_network.add_root(self.node1)
        self.tree_tensor_network.add_child_to_parent(self.node2, 0, "node1", 0)
        self.tree_tensor_network.add_child_to_parent(self.node3, 0, "node2", 1)
        self.tree_tensor_network.add_child_to_parent(self.node4, 0, "node2", 2)
        self.tree_tensor_network.add_child_to_parent(self.node5, 0, "node1", 1)
        self.tree_tensor_network.add_child_to_parent(self.node6, 0, "node5", 2)
        self.tree_tensor_network.add_child_to_parent(self.node7, 0, "node6", 1)

    def test_canonical_form_root(self):
        reference_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        ref_tensor = reference_ttn.nodes[reference_ttn.root_id].tensor
        
        ptn.canonical_form(self.tree_tensor_network, "node1")

        result_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        result_tensor = result_ttn.nodes[result_ttn.root_id].tensor

        self.assertTrue(np.allclose(ref_tensor,result_tensor))
        
        for node_id in self.tree_tensor_network.nodes:
            if node_id != self.tree_tensor_network.root_id:
                node = self.tree_tensor_network.nodes[node_id]
                tensor = node.tensor
                
                open_leg_indices = tuple(node.open_legs)
                children_leg_indices = tuple(node.children_legs.values())
                total_non_center_indices = open_leg_indices + children_leg_indices
                
                transfer_tensor = ptn.compute_transfer_tensor(tensor, total_non_center_indices)
                
                dimension_to_center = node.tensor.shape[node.parent_leg[1]]
                identity = np.eye(dimension_to_center)
                
                self.assertTrue(np.allclose(identity, transfer_tensor))

    def test_canoncial_form_non_root(self):
        reference_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        ref_tensor = reference_ttn.nodes[reference_ttn.root_id].tensor
        
        ptn.canonical_form(self.tree_tensor_network, "node2")

        result_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        result_tensor = result_ttn.nodes[result_ttn.root_id].tensor

        self.assertTrue(np.allclose(ref_tensor,result_tensor))
        
        for node_id in self.tree_tensor_network.nodes:
            if node_id == "node1":
                node = self.tree_tensor_network[node_id]
                tensor = node.tensor
                
                total_non_center_indices = tuple([node.children_legs[child_id] 
                                            for child_id in node.children_legs
                                            if child_id != "node2"])
                transfer_tensor = ptn.compute_transfer_tensor(tensor, total_non_center_indices)
                
                dimension_to_center = tensor.shape[node.children_legs["node2"]]
                identity = np.eye(dimension_to_center)
                
                self.assertTrue(np.allclose(identity, transfer_tensor))
                
            elif node_id != "node2":
                node = self.tree_tensor_network.nodes[node_id]
                tensor = node.tensor
                
                open_leg_indices = tuple(node.open_legs)
                children_leg_indices = tuple(node.children_legs.values())
                total_non_center_indices = open_leg_indices + children_leg_indices
                
                transfer_tensor = ptn.compute_transfer_tensor(tensor, total_non_center_indices)
                
                dimension_to_center = node.tensor.shape[node.parent_leg[1]]
                identity = np.eye(dimension_to_center)
                
                self.assertTrue(np.allclose(identity, transfer_tensor))
                    

if __name__ == "__main__":
    unittest.main()