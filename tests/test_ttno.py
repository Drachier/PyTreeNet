import unittest
import numpy as np

import pytreenet as ptn

class TestTTNOBasics(unittest.TestCase):

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

        ttno = ptn.TTNO.from_tensor(reference_tree, start_tensor, leg_dict)

        correct_open_leg_dimensions = {"id1": (2,6), "id2": (3,7),
                                        "id3": (4,8), "id4": (5,9)}

        for node_id in correct_open_leg_dimensions:
            correct_shape = correct_open_leg_dimensions[node_id]
            node = ttno.nodes[node_id]
            found_shape = node.shape_of_legs(node.open_legs)
            self.assertEqual(correct_shape, found_shape)

        contracted_ttno = ttno.completely_contract_tree(to_copy=True)
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

        ttno = ptn.TTNO.from_tensor(reference_tree, start_tensor, leg_dict)

        contracted_ttno = ttno.completely_contract_tree(to_copy=True)
        contracted_tensor = contracted_ttno.nodes[contracted_ttno.root_id].tensor

        # The shape is not retained throughout the entire procedure
        permutation = []
        self._find_permutation(ttno, leg_dict, ttno.root_id, permutation)

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

        ttno = ptn.TTNO.from_tensor(reference_tree, start_tensor, leg_dict)

        contracted_ttno = ttno.completely_contract_tree(to_copy=True)
        contracted_tensor = contracted_ttno.nodes[contracted_ttno.root_id].tensor

        # The shape is not retained throughout the entire procedure
        permutation = []
        self._find_permutation(ttno, leg_dict, ttno.root_id, permutation)

        correct_tensor = start_tensor.transpose(permutation)

        self.assertTrue(np.allclose(correct_tensor, contracted_tensor))

class TestTTNOfromHamiltonian(unittest.TestCase):
    def setUp(self):
        self.ref_tree = ptn.TreeTensorNetwork()

        node1 = ptn.TensorNode(ptn.crandn((2,2,2)), identifier="site1")
        node2 = ptn.TensorNode(ptn.crandn((2,2,2,2)), identifier="site2")
        node5 = ptn.TensorNode(ptn.crandn((2,2,2,2)), identifier="site5")
        node3 = ptn.TensorNode(ptn.crandn((2,2)), identifier="site3")
        node4 = ptn.TensorNode(ptn.crandn((2,2)), identifier="site4")
        node6 = ptn.TensorNode(ptn.crandn((2,2)), identifier="site6")
        node7 = ptn.TensorNode(ptn.crandn((2,2)), identifier="site7")

        self.ref_tree.add_root(node1)
        self.ref_tree.add_child_to_parent(node2, 0, "site1", 0)
        self.ref_tree.add_child_to_parent(node5, 0, "site1", 1)
        self.ref_tree.add_child_to_parent(node3, 0, "site2", 1)
        self.ref_tree.add_child_to_parent(node4, 0, "site2", 2)
        self.ref_tree.add_child_to_parent(node6, 0, "site5", 1)
        self.ref_tree.add_child_to_parent(node7, 0, "site5", 2)

        self.term = term = {"site1": "1",
                            "site2": "2",
                            "site3": "3",
                            "site4": "4",
                            "site5": "5",
                            "site6": "6",
                            "site7": "7"}

    def test_from_hamiltonian_one_term(self):
        conversion_dictionary = {"1": ptn.crandn((2,2)),
                                 "2": ptn.crandn((2,2)),
                                 "3": ptn.crandn((2,2)),
                                 "4": ptn.crandn((2,2)),
                                 "5": ptn.crandn((2,2)),
                                 "6": ptn.crandn((2,2)),
                                 "7": ptn.crandn((2,2)),
                                 }
        hamiltonian = ptn.Hamiltonian(terms=[self.term],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, self.ref_tree)

        for node_id in ttno.nodes:
            node = ttno.nodes[node_id]

            shape = node.tensor.shape
            # All open legs should have dimension 2
            for open_leg_index in node.open_legs:
                self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 1
            neighbours = node.neighbouring_nodes()
            for neighbour_id in neighbours:
                self.assertEqual(1, shape[neighbours[neighbour_id]])

    def test_from_hamiltonian_one_term_different_phys_dim(self):
        self.ref_tree.nodes["site2"].tensor = ptn.crandn((2,2,2,5))

        conversion_dictionary = {"1": ptn.crandn((2,2)),
                                 "2": ptn.crandn((5,5)),
                                 "3": ptn.crandn((2,2)),
                                 "4": ptn.crandn((2,2)),
                                 "5": ptn.crandn((2,2)),
                                 "6": ptn.crandn((2,2)),
                                 "7": ptn.crandn((2,2)),
                                 }
        hamiltonian = ptn.Hamiltonian(terms=[self.term],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, self.ref_tree)

        for node_id in ttno.nodes:
            node = ttno.nodes[node_id]

            shape = node.tensor.shape
            # Physical dimension at one site should be different
            for open_leg_index in node.open_legs:
                if node_id == "site2":
                    self.assertEqual(5, shape[open_leg_index])
                else:
                    self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 1
            neighbours = node.neighbouring_nodes()
            for neighbour_id in neighbours:
                self.assertEqual(1, shape[neighbours[neighbour_id]])

    def test_from_hamiltonian_two_terms_one_operator_different(self):
        conversion_dictionary = {"1": ptn.crandn((2,2)),
                                 "2": ptn.crandn((2,2)),
                                 "3": ptn.crandn((2,2)),
                                 "4": ptn.crandn((2,2)),
                                 "5": ptn.crandn((2,2)),
                                 "6": ptn.crandn((2,2)),
                                 "7": ptn.crandn((2,2)),
                                 "22": ptn.crandn((2,2))
                                 }
        term2 = {"site1": "1",
                 "site2": "22",
                 "site3": "3",
                 "site4": "4",
                 "site5": "5",
                 "site6": "6",
                 "site7": "7"}

        hamiltonian = ptn.Hamiltonian(terms=[self.term, term2],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, self.ref_tree)

        for node_id in ttno.nodes:
            node = ttno.nodes[node_id]

            shape = node.tensor.shape
            # All open legs should have dimension 2
            for open_leg_index in node.open_legs:
                self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 1
            neighbours = node.neighbouring_nodes()
            for neighbour_id in neighbours:
                self.assertEqual(1, shape[neighbours[neighbour_id]])

        # The operators at site 2 should be added
        ref_operator = conversion_dictionary["2"] + conversion_dictionary["22"]
        self.assertTrue(np.allclose(ref_operator, ttno.nodes["site2"].tensor))

    def test_from_hamiltonian_two_terms_completely_different(self):
        conversion_dictionary = {"1": ptn.crandn((2,2)),
                                 "2": ptn.crandn((2,2)),
                                 "3": ptn.crandn((2,2)),
                                 "4": ptn.crandn((2,2)),
                                 "5": ptn.crandn((2,2)),
                                 "6": ptn.crandn((2,2)),
                                 "7": ptn.crandn((2,2)),
                                 "12": ptn.crandn((2,2)),
                                 "22": ptn.crandn((2,2)),
                                 "32": ptn.crandn((2,2)),
                                 "42": ptn.crandn((2,2)),
                                 "52": ptn.crandn((2,2)),
                                 "62": ptn.crandn((2,2)),
                                 "72": ptn.crandn((2,2)),
                                 }
        term2 = {"site1": "12",
                 "site2": "22",
                 "site3": "32",
                 "site4": "42",
                 "site5": "52",
                 "site6": "62",
                 "site7": "72"}

        hamiltonian = ptn.Hamiltonian(terms=[self.term, term2],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, self.ref_tree)

        for node_id in ttno.nodes:
            node = ttno.nodes[node_id]

            shape = node.tensor.shape
            # All open legs should have dimension 2
            for open_leg_index in node.open_legs:
                self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 2
            neighbours = node.neighbouring_nodes()
            for neighbour_id in neighbours:
                self.assertEqual(2, shape[neighbours[neighbour_id]])

    def test_from_hamiltonian_two_terms_all_but_root_different(self):
        conversion_dictionary = {"1": ptn.crandn((2,2)),
                                 "2": ptn.crandn((2,2)),
                                 "3": ptn.crandn((2,2)),
                                 "4": ptn.crandn((2,2)),
                                 "5": ptn.crandn((2,2)),
                                 "6": ptn.crandn((2,2)),
                                 "7": ptn.crandn((2,2)),
                                 "22": ptn.crandn((2,2)),
                                 "32": ptn.crandn((2,2)),
                                 "42": ptn.crandn((2,2)),
                                 "52": ptn.crandn((2,2)),
                                 "62": ptn.crandn((2,2)),
                                 "72": ptn.crandn((2,2)),
                                 }
        term2 = {"site1": "1",
                 "site2": "22",
                 "site3": "32",
                 "site4": "42",
                 "site5": "52",
                 "site6": "62",
                 "site7": "72"}

        hamiltonian = ptn.Hamiltonian(terms=[self.term, term2],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, self.ref_tree)

        for node_id in ttno.nodes:
            node = ttno.nodes[node_id]

            shape = node.tensor.shape
            # All open legs should have dimension 2
            for open_leg_index in node.open_legs:
                self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 2
            neighbours = node.neighbouring_nodes()
            for neighbour_id in neighbours:
                self.assertEqual(2, shape[neighbours[neighbour_id]])

if __name__ == "__main__":
    unittest.main()