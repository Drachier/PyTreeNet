import unittest
import numpy as np
import copy
import pytreenet as ptn

from pytreenet.canonical_form import _find_smallest_distance_neighbour


class TestCanonicalFormSimple(unittest.TestCase):
    def setUp(self):
        self.tree_tensor_network = ptn.TreeTensorNetwork()

        self.node1, self.tensor1 = ptn.random_tensor_node((2, 3), identifier="node1")
        self.node2, self.tensor2 = ptn.random_tensor_node((2, 4, 5), identifier="node2")

        self.tree_tensor_network.add_root(self.node1, self.tensor1)
        self.tree_tensor_network.add_child_to_parent(self.node2, self.tensor2, 0, "node1", 0)

    def testsimple_find_smallest_distance_neighbour(self):
        distance_dict = self.tree_tensor_network.distance_to_node("node1")
        node2 = self.tree_tensor_network.nodes["node2"]
        minimum_distance_neighbour_id = _find_smallest_distance_neighbour(node2, distance_dict)
        self.assertEqual("node1", minimum_distance_neighbour_id)

    def testsimple_canonical_form(self):
        reference_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        ref_tensor = reference_ttn.tensors[reference_ttn.root_id]

        tensor1 = self.tree_tensor_network.tensors["node1"]
        tensor2 = self.tree_tensor_network.tensors["node2"]

        ref_tensor_direct = np.tensordot(tensor1, tensor2, axes=([0], [0]))
        self.assertTrue(np.allclose(ref_tensor, ref_tensor_direct))

        ptn.canonical_form(self.tree_tensor_network, "node1")

        result_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        result_tensor = result_ttn.tensors[result_ttn.root_id]

        self.assertFalse(result_ttn == reference_ttn)
        self.assertTrue(np.allclose(ref_tensor, result_tensor))

        node2, tensor2 = self.tree_tensor_network["node2"]
        parent_leg2 = node2.parent_leg
        parent_dimension2 = tensor2.shape[parent_leg2[1]]
        identity = np.eye(parent_dimension2)

        open_indices2 = node2.open_legs
        transfer_tensor = ptn.compute_transfer_tensor(tensor2, open_indices2)

        self.assertEqual(transfer_tensor.shape, (2, 2))
        self.assertTrue(np.allclose(identity, transfer_tensor))

class TestCanonicalFormComplicated(unittest.TestCase):
    def setUp(self):
        self.tree_tensor_network = ptn.TreeTensorNetwork()

        # Constructing a tree for tests
        self.node1, self.tensor1 = ptn.random_tensor_node((2, 3), identifier="node1")
        self.node2, self.tensor2 = ptn.random_tensor_node((2, 4, 5), identifier="node2")
        self.node3, self.tensor3 = ptn.random_tensor_node((4,), identifier="node3")
        self.node4, self.tensor4 = ptn.random_tensor_node((5, 8), identifier="node4")
        self.node5, self.tensor5 = ptn.random_tensor_node((3, 6, 7), identifier="node5")
        self.node6, self.tensor6 = ptn.random_tensor_node((7, 10, 9), identifier="node6")
        self.node7, self.tensor7 = ptn.random_tensor_node((10, 11, 12), identifier="node7")

        self.tree_tensor_network.add_root(self.node1, self.tensor1)
        self.tree_tensor_network.add_child_to_parent(self.node2, self.tensor2, 0, "node1", 0)
        self.tree_tensor_network.add_child_to_parent(self.node3, self.tensor3, 0, "node2", 1)
        self.tree_tensor_network.add_child_to_parent(self.node4, self.tensor4, 0, "node2", 2)
        self.tree_tensor_network.add_child_to_parent(self.node5, self.tensor5, 0, "node1", 1)
        self.tree_tensor_network.add_child_to_parent(self.node6, self.tensor6, 0, "node5", 2)
        self.tree_tensor_network.add_child_to_parent(self.node7, self.tensor7, 0, "node6", 1)

    def test_canonical_form_root(self):
        reference_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        ref_tensor = reference_ttn.tensors[reference_ttn.root_id]

        ptn.canonical_form(self.tree_tensor_network, "node1")

        result_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        result_tensor = result_ttn.tensors[result_ttn.root_id]

        self.assertTrue(np.allclose(ref_tensor, result_tensor))

        for node_id, tensor in self.tree_tensor_network.tensors.items():
            node = self.tree_tensor_network.nodes[node_id]
            if node.identifier != self.tree_tensor_network.root_id:

                open_leg_indices = tuple(node.open_legs)
                children_leg_indices = tuple(node.children_legs.values())
                total_non_center_indices = open_leg_indices + children_leg_indices

                transfer_tensor = ptn.compute_transfer_tensor(tensor, total_non_center_indices)

                dimension_to_center = tensor.shape[node.parent_leg[1]]
                identity = np.eye(dimension_to_center)

                self.assertTrue(np.allclose(identity, transfer_tensor))

    def _check_node(self, ttn, node, id_neighbour_towards_center):
        node_neighbour_legs = node.neighbouring_nodes()

        non_center_indices = list(range(node.nlegs()))
        non_center_indices.pop(node.get_neighbour_leg(id_neighbour_towards_center))

        tensor = ttn.tensors[node.identifier]
        transfer_tensor = ptn.compute_transfer_tensor(tensor, non_center_indices)

        # Dimension of leg towards the neighbour nearest to the orth. center
        dimension_to_center = tensor.shape[node.get_neighbour_leg(id_neighbour_towards_center)]

        identity = np.eye(dimension_to_center)

        self.assertTrue(np.allclose(identity, transfer_tensor))

        for neighbour_id in node_neighbour_legs:
            if neighbour_id != id_neighbour_towards_center:
                self._check_node(ttn, ttn.nodes[neighbour_id], node.identifier)

    def test_canoncial_form_non_root(self):
        reference_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        ref_tensor = reference_ttn.tensors[reference_ttn.root_id]

        # We can find the scalar product of this TTN with itself, by contracting
        # all legs left.
        ref_scalar_product = ptn.compute_transfer_tensor(ref_tensor, range(ref_tensor.ndim))

        for node_id_center in self.tree_tensor_network.nodes:
            canon_ttn = copy.deepcopy(self.tree_tensor_network)

            ptn.canonical_form(canon_ttn, node_id_center)

            # Test, if both TTN represent the same tensor network
            result_ttn = ptn.completely_contract_tree(canon_ttn, to_copy=True)
            result_tensor = result_ttn.tensors[result_ttn.root_id]
            perm = []
            for i in result_tensor.shape:
                perm.append(ref_tensor.shape.index(i))
            ref_tensor2 = ref_tensor.transpose(perm)

            self.assertTrue(np.allclose(ref_tensor2, result_tensor))

            # Test, if we actually have a canonical form
            center_node = canon_ttn.nodes[node_id_center]

            # Contracting this node's tensor with the conjugate should give the
            # same result as the total scalar product.
            scalar_product = ptn.compute_transfer_tensor(ref_tensor,
                                                         range(ref_tensor.ndim))

            self.assertTrue(np.allclose(ref_scalar_product, scalar_product))

            # All other tensors should become the identity, when contracting all
            # non_center pointing legs with the complex conjugate
            neighbours_of_center = center_node.neighbouring_nodes()

            for neighbour_id in neighbours_of_center:
                self._check_node(canon_ttn,
                                 canon_ttn.nodes[neighbour_id],
                                 node_id_center)


if __name__ == "__main__":
    unittest.main()
