import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn
from pytreenet.random import (random_tensor_node,
                              crandn)

class TestTTNOBasics(unittest.TestCase):

    def _find_permutation_rec(self, ttno, leg_dict, node_id, perm):
        node, _ = ttno[node_id]
        input_index = leg_dict[node_id]
        output_index = input_index + len(ttno.nodes)
        perm.extend([input_index, output_index])
        if not node.is_leaf():
            for child_id in node.children:
                self._find_permutation_rec(ttno, leg_dict, child_id, perm)

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

        node1, tensor1 = random_tensor_node((d, d, d), identifier="id1")
        node2, tensor2 = random_tensor_node((d, d), identifier="id2")
        node3, tensor3 = random_tensor_node((d, d, d), identifier="id3")
        node4, tensor4 = random_tensor_node((d, d), identifier="id4")

        reference_tree.add_root(node1, tensor1)
        reference_tree.add_child_to_parent(node2, tensor2, 1, "id1", 2)
        reference_tree.add_child_to_parent(node3, tensor3,  1, "id1", 1)
        reference_tree.add_child_to_parent(node4, tensor4,  1, "id3", 2)

        shape = [2, 3, 4, 5, 6, 7, 8, 9]
        start_tensor = crandn(shape)
        leg_dict = {"id1": 0, "id2": 1, "id3": 2, "id4": 3}

        ttno = ptn.TTNO.from_tensor(reference_tree, start_tensor, leg_dict)

        correct_open_leg_dimensions = {"id1": (2, 6), "id2": (3, 7),
                                       "id3": (4, 8), "id4": (5, 9)}

        for node_id, correct_shape in correct_open_leg_dimensions.items():
            tensor = ttno.tensors[node_id]
            found_shape = tensor.shape[-2:]
            self.assertEqual(correct_shape, found_shape)

        contracted_tensor = ttno.completely_contract_tree(to_copy=True)[0]
        # The shape is not retained throughout the entire procedure
        correct_tensor = start_tensor.transpose((0, 4, 1, 5, 2, 6, 3, 7))

        self.assertTrue(np.allclose(correct_tensor, contracted_tensor))

    def test_from_tensorB(self):
        reference_tree = ptn.TreeTensorNetwork()
        node1, tensor1 = random_tensor_node((2, 3, 4, 5), identifier="id1")
        node2, tensor2 = random_tensor_node((2, 3, 4, 5), identifier="id2")
        node3, tensor3 = random_tensor_node((2, 3, 4, 5), identifier="id3")
        node4, tensor4 = random_tensor_node((2, 3, 4, 5), identifier="id4")
        node5, tensor5 = random_tensor_node((2, 3, 4, 5), identifier="id5")

        reference_tree.add_root(node1, tensor1)
        reference_tree.add_child_to_parent(node2, tensor2, 0, "id1", 0)
        reference_tree.add_child_to_parent(node3, tensor3, 1, "id1", 1)
        reference_tree.add_child_to_parent(node4, tensor4, 2, "id3", 2)
        reference_tree.add_child_to_parent(node5, tensor5, 2, "id1", 2)

        num_nodes = len(reference_tree.nodes)

        shape = [i for i in range(2, (num_nodes + 2))]
        shape.extend(shape)
        start_tensor = crandn(shape)
        leg_dict = {"id" + str(i + 1): i for i in range(num_nodes)}
        ttno = ptn.TTNO.from_tensor(reference_tree, start_tensor, leg_dict)

        contracted_tensor = ttno.completely_contract_tree(to_copy=True)[0]
        # The shape is not retained throughout the entire procedure
        permutation = []
        self._find_permutation_rec(ttno, leg_dict, ttno.root_id, permutation)
        correct_tensor = start_tensor.transpose(permutation)

        self.assertTrue(np.allclose(correct_tensor, contracted_tensor))

    def test_from_tensorC(self):
        reference_tree = ptn.TreeTensorNetwork()
        node1, tensor1 = random_tensor_node((2, 3, 4, 5), identifier="id1")
        node2, tensor2 = random_tensor_node((2, 3, 4, 5), identifier="id2")
        node3, tensor3 = random_tensor_node((2, 3, 4, 5), identifier="id3")
        node4, tensor4 = random_tensor_node((2, 3, 4, 5), identifier="id4")
        node5, tensor5 = random_tensor_node((2, 3, 4, 5), identifier="id5")
        node6, tensor6 = random_tensor_node((2, 3, 4, 5), identifier="id6")
        node7, tensor7 = random_tensor_node((2, 3, 4, 5), identifier="id7")
        node8, tensor8 = random_tensor_node((2, 3, 4, 5), identifier="id8")
        node9, tensor9 = random_tensor_node((2, 3, 4, 5), identifier="id9")
        node10, tensor10 = random_tensor_node((2, 3, 4, 5), identifier="id10")

        reference_tree.add_root(node1, tensor1)
        reference_tree.add_child_to_parent(node2, tensor2, 0, "id1", 0)
        reference_tree.add_child_to_parent(node3, tensor3, 3, "id2", 3)
        reference_tree.add_child_to_parent(node4, tensor4, 0, "id3", 1)
        reference_tree.add_child_to_parent(node5, tensor5, 2, "id2", 3)
        reference_tree.add_child_to_parent(node6, tensor6, 0, "id5", 1)
        reference_tree.add_child_to_parent(node7, tensor7, 1, "id5", 2)
        reference_tree.add_child_to_parent(node8, tensor8, 1, "id1", 1)
        reference_tree.add_child_to_parent(node9, tensor9, 2, "id8", 2)
        reference_tree.add_child_to_parent(node10, tensor10, 2, "id1", 2)

        d = 2
        num_nodes = len(reference_tree.nodes)

        shape = [d for i in range(2 * num_nodes)]
        start_tensor = crandn(shape)
        leg_dict = {"id" + str(i + 1): i for i in range(num_nodes)}
        ttno = ptn.TTNO.from_tensor(reference_tree, start_tensor, leg_dict)

        contracted_tensor = ttno.completely_contract_tree(to_copy=True)[0]

        # The shape is not retained throughout the entire procedure
        permutation = []
        self._find_permutation_rec(ttno, leg_dict, ttno.root_id, permutation)
        correct_tensor = start_tensor.transpose(permutation)
        self.assertTrue(np.allclose(correct_tensor, contracted_tensor))

    def test_from_tensor_root_is_leaf(self):
        reference_tree = ptn.TreeTensorNetworkState()
        shapes = [(1,1,1,2),(1,1,1,2),(1,2),(1,2),(1,2),(1,1,2),(1,2)]
        nodes = [random_tensor_node(shape, "node" +str(i))
                 for i, shape in enumerate(shapes)]
        # Build tree
        reference_tree.add_root(nodes[4][0],nodes[4][1])
        reference_tree.add_child_to_parent(nodes[0][0], nodes[0][1],0,"node4",0)
        reference_tree.add_child_to_parent(nodes[1][0], nodes[1][1],0,"node0",1)
        reference_tree.add_child_to_parent(nodes[2][0], nodes[2][1],0,"node1",1)
        reference_tree.add_child_to_parent(nodes[3][0], nodes[3][1],0,"node1",2)
        reference_tree.add_child_to_parent(nodes[5][0], nodes[5][1],0,"node0",2)
        reference_tree.add_child_to_parent(nodes[6][0], nodes[6][1],0,"node5",1)

        shape = 2*len(nodes)*[2]
        start_tensor = crandn(shape)
        leg_dict = {"node"+str(i): i for i in range(len(nodes))}
        ttno = ptn.TTNO.from_tensor(reference_tree, deepcopy(start_tensor), leg_dict)

        contracted_tensor = ttno.completely_contract_tree(to_copy=True)[0]
        # The shape is not retained throughout the entire procedure
        permutation = []
        self._find_permutation_rec(ttno, leg_dict, ttno.root_id, permutation)
        correct_tensor = start_tensor.transpose(permutation)
        self.assertTrue(np.allclose(correct_tensor, contracted_tensor))

class TestTTNOfromHamiltonian(unittest.TestCase):
    def setUp(self):
        self.ref_tree = ptn.TreeTensorNetwork()

        node1, tensor1 = random_tensor_node((2, 2, 2), identifier="site1")
        node2, tensor2 = random_tensor_node((2, 2, 2, 2), identifier="site2")
        node5, tensor5 = random_tensor_node((2, 2, 2, 2), identifier="site5")
        node3, tensor3 = random_tensor_node((2, 2), identifier="site3")
        node4, tensor4 = random_tensor_node((2, 2), identifier="site4")
        node6, tensor6 = random_tensor_node((2, 2), identifier="site6")
        node7, tensor7 = random_tensor_node((2, 2), identifier="site7")

        self.ref_tree.add_root(node1, tensor1)
        self.ref_tree.add_child_to_parent(node2, tensor2, 0, "site1", 0)
        self.ref_tree.add_child_to_parent(node5, tensor5, 0, "site1", 1)
        self.ref_tree.add_child_to_parent(node3, tensor3, 0, "site2", 1)
        self.ref_tree.add_child_to_parent(node4, tensor4, 0, "site2", 2)
        self.ref_tree.add_child_to_parent(node6, tensor6, 0, "site5", 1)
        self.ref_tree.add_child_to_parent(node7, tensor7, 0, "site5", 2)

        self.term = ptn.TensorProduct({"site1": "1",
                                        "site2": "2",
                                        "site3": "3",
                                        "site4": "4",
                                        "site5": "5",
                                        "site6": "6",
                                        "site7": "7"})

        self.transpose_permutation = (7,0,8,1,10,3,11,4,9,2,12,5,13,6)

    def test_from_hamiltonian_one_term(self):
        conversion_dictionary = {"1": crandn((2, 2)),
                                 "2": crandn((2, 2)),
                                 "3": crandn((2, 2)),
                                 "4": crandn((2, 2)),
                                 "5": crandn((2, 2)),
                                 "6": crandn((2, 2)),
                                 "7": crandn((2, 2)),
                                 }
        hamiltonian = ptn.Hamiltonian(terms=[self.term],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, self.ref_tree)
        for node_id in ttno.nodes:
            node, tensor = ttno[node_id]

            shape = tensor.shape
            # All open legs should have dimension 2
            for open_leg_index in node.open_legs:
                self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 1
            for children_leg in node.children_legs:
                self.assertEqual(1, shape[children_leg])

            if node.nparents() != 0:
                self.assertEqual(1, shape[node.parent_leg])

        hamiltonian_tensor = hamiltonian.to_tensor(self.ref_tree).operator.transpose(self.transpose_permutation)
        found_tensor = ttno.completely_contract_tree()[0]
        self.assertTrue(np.allclose(hamiltonian_tensor, found_tensor))
        
    def test_from_hamiltonian_one_term_real(self):
        conversion_dictionary = {"1": np.random.randn(2, 2),
                                 "2": np.random.randn(2, 2),
                                 "3": np.random.randn(2, 2),
                                 "4": np.random.randn(2, 2),
                                 "5": np.random.randn(2, 2),
                                 "6": np.random.randn(2, 2),
                                 "7": np.random.randn(2, 2),
                                 }
        hamiltonian = ptn.Hamiltonian(terms=[self.term],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, self.ref_tree, dtype=float)
        for node_id in ttno.nodes:
            node, tensor = ttno[node_id]

            shape = tensor.shape
            # All open legs should have dimension 2
            for open_leg_index in node.open_legs:
                self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 1
            for children_leg in node.children_legs:
                self.assertEqual(1, shape[children_leg])

            if node.nparents() != 0:
                self.assertEqual(1, shape[node.parent_leg])

        hamiltonian_tensor = hamiltonian.to_tensor(self.ref_tree).operator.transpose(self.transpose_permutation)
        found_tensor = ttno.completely_contract_tree()[0]
        self.assertTrue(np.allclose(hamiltonian_tensor, found_tensor))

    def test_from_hamiltonian_one_term_different_phys_dim(self):
        ref_tree = ptn.TreeTensorNetwork()

        node1, tensor1 = random_tensor_node((2, 2, 2), identifier="site1")
        node2, tensor2 = random_tensor_node((2, 2, 2, 5), identifier="site2")
        node5, tensor5 = random_tensor_node((2, 2, 2, 2), identifier="site5")
        node3, tensor3 = random_tensor_node((2, 2), identifier="site3")
        node4, tensor4 = random_tensor_node((2, 2), identifier="site4")
        node6, tensor6 = random_tensor_node((2, 2), identifier="site6")
        node7, tensor7 = random_tensor_node((2, 2), identifier="site7")

        ref_tree.add_root(node1, tensor1)
        ref_tree.add_child_to_parent(node2, tensor2, 0, "site1", 0)
        ref_tree.add_child_to_parent(node5, tensor5, 0, "site1", 1)
        ref_tree.add_child_to_parent(node3, tensor3, 0, "site2", 1)
        ref_tree.add_child_to_parent(node4, tensor4, 0, "site2", 2)
        ref_tree.add_child_to_parent(node6, tensor6, 0, "site5", 1)
        ref_tree.add_child_to_parent(node7, tensor7, 0, "site5", 2)

        conversion_dictionary = {"1": crandn((2, 2)),
                                 "2": crandn((5, 5)),
                                 "3": crandn((2, 2)),
                                 "4": crandn((2, 2)),
                                 "5": crandn((2, 2)),
                                 "6": crandn((2, 2)),
                                 "7": crandn((2, 2)),
                                 }
        hamiltonian = ptn.Hamiltonian(terms=[self.term],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, ref_tree)

        for node_id in ttno.nodes:
            node, tensor = ttno[node_id]

            shape = tensor.shape
            # Physical dimension at one site should be different
            for open_leg_index in node.open_legs:
                if node_id == "site2":
                    self.assertEqual(5, shape[open_leg_index])
                else:
                    self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 1
            for children_leg in node.children_legs:
                self.assertEqual(1, shape[children_leg])

            if node.nparents() != 0:
                self.assertEqual(1, shape[node.parent_leg])

        hamiltonian_tensor = hamiltonian.to_tensor(ref_tree).operator.transpose(self.transpose_permutation)
        found_tensor = ttno.completely_contract_tree()[0]
        self.assertTrue(np.allclose(hamiltonian_tensor, found_tensor))

    def test_from_hamiltonian_two_terms_one_operator_different(self):
        conversion_dictionary = {"1": crandn((2, 2)),
                                 "2": crandn((2, 2)),
                                 "3": crandn((2, 2)),
                                 "4": crandn((2, 2)),
                                 "5": crandn((2, 2)),
                                 "6": crandn((2, 2)),
                                 "7": crandn((2, 2)),
                                 "22": crandn((2, 2))
                                 }
        term2 = ptn.TensorProduct({"site1": "1",
                                    "site2": "22",
                                    "site3": "3",
                                    "site4": "4",
                                    "site5": "5",
                                    "site6": "6",
                                    "site7": "7"})

        hamiltonian = ptn.Hamiltonian(terms=[self.term, term2],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, self.ref_tree)

        for node_id in ttno.nodes:
            node, tensor = ttno[node_id]

            shape = tensor.shape
            # All open legs should have dimension 2
            for open_leg_index in node.open_legs:
                self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 1
            for children_leg in node.children_legs:
                self.assertEqual(1, shape[children_leg])

            if node.nparents() != 0:
                self.assertEqual(1, shape[node.parent_leg])

        # The operators at site 2 should be added
        ref_operator = conversion_dictionary["2"] + conversion_dictionary["22"]
        self.assertTrue(np.allclose(ref_operator, ttno.tensors["site2"]))

        # The TTNO and Hamiltonian should be equal
        hamiltonian_tensor = hamiltonian.to_tensor(self.ref_tree).operator.transpose(self.transpose_permutation)
        found_tensor = ttno.completely_contract_tree()[0]
        self.assertTrue(np.allclose(hamiltonian_tensor, found_tensor))

    def test_from_hamiltonian_two_terms_completely_different(self):
        conversion_dictionary = {"1": crandn((2, 2)),
                                 "2": crandn((2, 2)),
                                 "3": crandn((2, 2)),
                                 "4": crandn((2, 2)),
                                 "5": crandn((2, 2)),
                                 "6": crandn((2, 2)),
                                 "7": crandn((2, 2)),
                                 "12": crandn((2, 2)),
                                 "22": crandn((2, 2)),
                                 "32": crandn((2, 2)),
                                 "42": crandn((2, 2)),
                                 "52": crandn((2, 2)),
                                 "62": crandn((2, 2)),
                                 "72": crandn((2, 2)),
                                 }
        term2 = ptn.TensorProduct({"site1": "12",
                                    "site2": "22",
                                    "site3": "32",
                                    "site4": "42",
                                    "site5": "52",
                                    "site6": "62",
                                    "site7": "72"})

        hamiltonian = ptn.Hamiltonian(terms=[self.term, term2],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, self.ref_tree)

        for node_id in ttno.nodes:
            node, tensor = ttno[node_id]

            shape = tensor.shape
            # All open legs should have dimension 2
            for open_leg_index in node.open_legs:
                self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 2
            for children_leg in node.children_legs:
                self.assertEqual(2, shape[children_leg])

            if node.nparents() != 0:
                self.assertEqual(2, shape[node.parent_leg])
    
        # The TTNO and Hamiltonian should be equal
        hamiltonian_tensor = hamiltonian.to_tensor(self.ref_tree).operator.transpose(self.transpose_permutation)
        found_tensor = ttno.completely_contract_tree()[0]
        self.assertTrue(np.allclose(hamiltonian_tensor, found_tensor))
        
    def test_from_hamiltonian_two_terms_completely_different_real(self):
        conversion_dictionary = {"1": np.random.randn(2, 2),
                                 "2": np.random.randn(2, 2),
                                 "3": np.random.randn(2, 2),
                                 "4": np.random.randn(2, 2),
                                 "5": np.random.randn(2, 2),
                                 "6": np.random.randn(2, 2),
                                 "7": np.random.randn(2, 2),
                                 "12": np.random.randn(2, 2),
                                 "22": np.random.randn(2, 2),
                                 "32": np.random.randn(2, 2),
                                 "42": np.random.randn(2, 2),
                                 "52": np.random.randn(2, 2),
                                 "62": np.random.randn(2, 2),
                                 "72": np.random.randn(2, 2),
                                 }
        term2 = ptn.TensorProduct({"site1": "12",
                                    "site2": "22",
                                    "site3": "32",
                                    "site4": "42",
                                    "site5": "52",
                                    "site6": "62",
                                    "site7": "72"})

        hamiltonian = ptn.Hamiltonian(terms=[self.term, term2],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, self.ref_tree, dtype=float)

        for node_id in ttno.nodes:
            node, tensor = ttno[node_id]

            shape = tensor.shape
            # All open legs should have dimension 2
            for open_leg_index in node.open_legs:
                self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 2
            for children_leg in node.children_legs:
                self.assertEqual(2, shape[children_leg])

            if node.nparents() != 0:
                self.assertEqual(2, shape[node.parent_leg])
    
        # The TTNO and Hamiltonian should be equal
        hamiltonian_tensor = hamiltonian.to_tensor(self.ref_tree).operator.transpose(self.transpose_permutation)
        found_tensor = ttno.completely_contract_tree()[0]
        self.assertTrue(np.allclose(hamiltonian_tensor, found_tensor))

    def test_from_hamiltonian_two_terms_all_but_root_different(self):
        conversion_dictionary = {"1": crandn((2, 2)),
                                 "2": crandn((2, 2)),
                                 "3": crandn((2, 2)),
                                 "4": crandn((2, 2)),
                                 "5": crandn((2, 2)),
                                 "6": crandn((2, 2)),
                                 "7": crandn((2, 2)),
                                 "22": crandn((2, 2)),
                                 "32": crandn((2, 2)),
                                 "42": crandn((2, 2)),
                                 "52": crandn((2, 2)),
                                 "62": crandn((2, 2)),
                                 "72": crandn((2, 2)),
                                 }
        term2 = ptn.TensorProduct({"site1": "1",
                                    "site2": "22",
                                    "site3": "32",
                                    "site4": "42",
                                    "site5": "52",
                                    "site6": "62",
                                    "site7": "72"})

        hamiltonian = ptn.Hamiltonian(terms=[self.term, term2],
                                      conversion_dictionary=conversion_dictionary)

        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, self.ref_tree)

        for node_id in ttno.nodes:
            node, tensor = ttno[node_id]

            shape = tensor.shape
            # All open legs should have dimension 2
            for open_leg_index in node.open_legs:
                self.assertEqual(2, shape[open_leg_index])

            # All other legs should have dimension 2
            for children_leg in node.children_legs:
                self.assertEqual(2, shape[children_leg])

            if node.nparents() != 0:
                self.assertEqual(2, shape[node.parent_leg])

        # The TTNO and Hamiltonian should be equal
        hamiltonian_tensor = hamiltonian.to_tensor(self.ref_tree).operator.transpose(self.transpose_permutation)
        found_tensor = ttno.completely_contract_tree()[0]
        self.assertTrue(np.allclose(hamiltonian_tensor, found_tensor))


if __name__ == "__main__":
    unittest.main()
