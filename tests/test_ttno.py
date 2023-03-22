import unittest
import numpy as np

import pytreenet as ptn

class TestTTNOBasics(unittest.TestCase):

    def setUp(self):
        self.ttno = ptn.TTNO()

    def _find_permutation_rec(self, ttno, leg_dict, node_id, perm):
        """
        Working Code, but for a wrong contraction order. Might get usefull once
        we rework contract_completely to work recursively
        """
        node = ttno.nodes[node_id]

        input_index = leg_dict[node_id]
        output_index = input_index + len(ttno.nodes)
        perm.extend([input_index, output_index])

        if not node.is_leaf():
            for child_id in node.children_legs:
                self._find_permutation(ttno, leg_dict, child_id, perm)

    def _find_permutation(self, ttno, leg_dict, node_id, perm):
        distance_to_root = ttno.distance_to_node(ttno.root_id)

        input_index = leg_dict[node_id]
        output_index = input_index + len(ttno.nodes)
        perm.extend([input_index, output_index])

        for distance in range(1, max(distance_to_root.values())+1):
                node_id_with_distance = [node_id for node_id in distance_to_root
                                         if distance_to_root[node_id] == distance]
                for node_id in node_id_with_distance:
                    input_index = leg_dict[node_id]
                    output_index = input_index + len(ttno.nodes)
                    perm.extend([input_index, output_index])

    def test_from_tensorA(self):
        reference_tree = ptn.TreeTensorNetwork()
        d = 2

        node1 = ptn.random_tensor_node((d,d,d), identifier="id1")
        node2 = ptn.random_tensor_node((d,d), identifier="id2")
        node3 = ptn.random_tensor_node((d,d,d), identifier="id3")
        node4 = ptn.random_tensor_node((d,d), identifier="id4")

        reference_tree.add_root(node1)
        reference_tree.add_child_to_parent(node2, 1, "id1", 2)
        reference_tree.add_child_to_parent(node3, 1, "id1", 1)
        reference_tree.add_child_to_parent(node4, 1, "id3", 2)

        shape = [2,3,4,5,6,7,8,9]
        start_tensor = ptn.crandn(shape)
        leg_dict = {"id1": 0, "id2": 1, "id3": 2, "id4": 3}

        self.ttno.from_tensor(reference_tree, start_tensor, leg_dict)

        correct_open_leg_dimensions = {"id1": (2,6), "id2": (3,7),
                                        "id3": (4,8), "id4": (5,9)}

        for node_id in correct_open_leg_dimensions:
            correct_shape = correct_open_leg_dimensions[node_id]
            node = self.ttno.nodes[node_id]
            found_shape = node.shape_of_legs(node.open_legs)
            self.assertEqual(correct_shape, found_shape)

        contracted_ttno = self.ttno.completely_contract_tree(to_copy=True)
        contracted_tensor = contracted_ttno.nodes[contracted_ttno.root_id].tensor

        # The shape is not retained throughout the entire procedure
        correct_tensor = start_tensor.transpose((0,4,1,5,2,6,3,7))

        self.assertTrue(np.allclose(correct_tensor, contracted_tensor))

    def test_from_tensorB(self):
        reference_tree = ptn.TreeTensorNetwork()
        node1 = ptn.random_tensor_node((2,3,4,5), identifier="id1")
        node2 = ptn.random_tensor_node((2,3,4,5), identifier="id2")
        node3 = ptn.random_tensor_node((2,3,4,5), identifier="id3")
        node4 = ptn.random_tensor_node((2,3,4,5), identifier="id4")
        node5 = ptn.random_tensor_node((2,3,4,5), identifier="id5")

        reference_tree.add_root(node1)
        reference_tree.add_child_to_parent(node2, 0, "id1", 0)
        reference_tree.add_child_to_parent(node3, 1, "id1", 1)
        reference_tree.add_child_to_parent(node4, 2, "id3", 2)
        reference_tree.add_child_to_parent(node5, 2, "id1", 2)

        num_nodes = len(reference_tree.nodes)

        shape = [i for i in range(2,(num_nodes + 2))]
        shape.extend(shape)
        start_tensor = ptn.crandn(shape)
        leg_dict = {"id" + str(i + 1): i for i in range(num_nodes)}

        self.ttno.from_tensor(reference_tree, start_tensor, leg_dict)

        contracted_ttno = self.ttno.completely_contract_tree(to_copy=True)
        contracted_tensor = contracted_ttno.nodes[contracted_ttno.root_id].tensor

        # The shape is not retained throughout the entire procedure
        permutation = []
        self._find_permutation(self.ttno, leg_dict, self.ttno.root_id, permutation)

        correct_tensor = start_tensor.transpose(permutation)

        self.assertTrue(np.allclose(correct_tensor, contracted_tensor))

    def test_from_tensorC(self):
        reference_tree = ptn.TreeTensorNetwork()
        node1 = ptn.random_tensor_node((2,3,4,5), identifier="id1")
        node2 = ptn.random_tensor_node((2,3,4,5), identifier="id2")
        node3 = ptn.random_tensor_node((2,3,4,5), identifier="id3")
        node4 = ptn.random_tensor_node((2,3,4,5), identifier="id4")
        node5 = ptn.random_tensor_node((2,3,4,5), identifier="id5")
        node6 = ptn.random_tensor_node((2,3,4,5), identifier="id6")
        node7 = ptn.random_tensor_node((2,3,4,5), identifier="id7")
        node8 = ptn.random_tensor_node((2,3,4,5), identifier="id8")
        node9 = ptn.random_tensor_node((2,3,4,5), identifier="id9")
        node10 = ptn.random_tensor_node((2,3,4,5), identifier="id10")

        reference_tree.add_root(node1)
        reference_tree.add_child_to_parent(node2, 0, "id1", 0)
        reference_tree.add_child_to_parent(node3, 3, "id2", 3)
        reference_tree.add_child_to_parent(node4, 0, "id3", 0)
        reference_tree.add_child_to_parent(node5, 2, "id2", 2)
        reference_tree.add_child_to_parent(node6, 0, "id5", 0)
        reference_tree.add_child_to_parent(node7, 1, "id5", 1)
        reference_tree.add_child_to_parent(node8, 1, "id1", 1)
        reference_tree.add_child_to_parent(node9, 2, "id8", 2)
        reference_tree.add_child_to_parent(node10, 2, "id1", 2)

        d = 2
        num_nodes = len(reference_tree.nodes)

        shape = [d for i in range(2 * num_nodes)]
        start_tensor = ptn.crandn(shape)
        leg_dict = {"id" + str(i + 1): i for i in range(num_nodes)}

        self.ttno.from_tensor(reference_tree, start_tensor, leg_dict)

        contracted_ttno = self.ttno.completely_contract_tree(to_copy=True)
        contracted_tensor = contracted_ttno.nodes[contracted_ttno.root_id].tensor

        # The shape is not retained throughout the entire procedure
        permutation = []
        self._find_permutation(self.ttno, leg_dict, self.ttno.root_id, permutation)

        correct_tensor = start_tensor.transpose(permutation)

        self.assertTrue(np.allclose(correct_tensor, contracted_tensor))

if __name__ == "__main__":
    unittest.main()
