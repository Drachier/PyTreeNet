import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn


class TestTreeTensorNetworkBasics(unittest.TestCase):

    def setUp(self):
        self.tensortree = ptn.TreeTensorNetwork()
        self.node1, self.tensor1 = ptn.random_tensor_node((2, 3, 4, 5), identifier="orig_root")
        self.node2, self.tensor2 = ptn.random_tensor_node((2, 3), identifier="child1")
        self.node3, self.tensor3 = ptn.random_tensor_node((2, 3, 4, 5), identifier="child2")
        self.node4, self.tensor4 = ptn.random_tensor_node((2, 3, 4, 5), identifier="new_root")

    def test_add_root(self):
        self.assertEqual(self.tensortree.root_id, None)
        self.assertEqual(self.tensortree.nodes, {})

        self.tensortree.add_root(self.node1, self.tensor1)

        self.assertEqual(self.tensortree.root_id, "orig_root")
        self.assertEqual(len(self.tensortree.nodes), 1)
        self.assertEqual(len(self.tensortree.tensors), 1)

    def test_add_child_to_parent(self):
        self.tensortree.add_root(self.node1, self.tensor1)

        # Depth 2
        self.tensortree.add_child_to_parent(self.node2, self.tensor2, 1, "orig_root", 1)
        self.assertEqual(len(self.tensortree.nodes), 2)
        permutation = self.tensortree.nodes["child1"].leg_permutation
        transposed_tensor = self.tensor2.transpose(permutation)
        self.assertTrue(np.allclose(self.tensortree.tensors["child1"], transposed_tensor))
        self.assertEqual(self.tensortree.nodes["child1"].parent, "orig_root")
        self.assertEqual(self.tensortree.nodes["child1"].children, [])
        self.assertEqual(self.tensortree.nodes["orig_root"].children, ["child1"])

        # Depth 3
        self.tensortree.add_child_to_parent(self.node3, self.tensor3, 0, "child1", 0)
        self.assertEqual(len(self.tensortree.nodes), 3)
        permutation = self.tensortree.nodes["child2"].leg_permutation
        transposed_tensor = self.tensor3.transpose(permutation)
        self.assertTrue(np.allclose(self.tensortree.tensors["child2"], transposed_tensor))
        self.assertEqual(self.tensortree.nodes["child2"].parent, "child1")
        self.assertEqual(self.tensortree.nodes["child2"].children, [])
        self.assertEqual(self.tensortree.nodes["child1"].children, ["child2"])

    def test_add_parent_to_root(self):
        # Setup
        self.tensortree.add_root(self.node1, self.tensor1)
        self.tensortree.add_child_to_parent(self.node2, self.tensor2, 1, "orig_root", 1)
        self.tensortree.add_child_to_parent(self.node3, self.tensor3, 0, "child1", 0)

        self.tensortree.add_parent_to_root(0, self.node4, self.tensor4, 0)
        self.assertEqual(self.tensortree.root_id, "new_root")
        self.assertEqual(len(self.tensortree.nodes), 4)
        permutation = self.tensortree.nodes["new_root"].leg_permutation
        transposed_tensor = self.tensor4.transpose(permutation)
        self.assertTrue(np.allclose(self.tensortree.tensors["new_root"], transposed_tensor))
        self.assertEqual(self.tensortree.nodes["new_root"].parent, None)
        self.assertEqual(self.tensortree.nodes["new_root"].children, ["orig_root"])
        self.assertEqual(self.tensortree.nodes["orig_root"].parent, "new_root")


class TestTreeTensorNetworkBigTree(unittest.TestCase):
    def setUp(self):
        self.tensortree = ptn.TreeTensorNetwork()

        node1, tensor1 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id1")
        node2, tensor2 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id2")
        node3, tensor3 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id3")
        node4, tensor4 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id4")
        node5, tensor5 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id5")
        node6, tensor6 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id6")
        node7, tensor7 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id7")
        node8, tensor8 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id8")
        node9, tensor9 = ptn.random_tensor_node((2, 3, 4, 5), identifier="id9")

        self.tensortree.add_root(node1, tensor1)
        self.tensortree.add_child_to_parent(node2, tensor2, 0, "id1", 0)
        self.tensortree.add_child_to_parent(node3, tensor3, 3, "id2", 3)
        self.tensortree.add_child_to_parent(node4, tensor4, 0, "id3", 0)
        self.tensortree.add_child_to_parent(node5, tensor5, 2, "id2", 2)
        self.tensortree.add_child_to_parent(node6, tensor6, 0, "id5", 0)
        self.tensortree.add_child_to_parent(node7, tensor7, 1, "id5", 1)
        self.tensortree.add_child_to_parent(node8, tensor8, 1, "id1", 1)
        self.tensortree.add_child_to_parent(node9, tensor9, 2, "id8", 2)

    def test_absorb_into_open_legs(self):
        tensor_shape_dict = {"id1": (4, 5, 4, 5),
                             "id2": (3, 3),
                             "id3": (3, 4, 3, 4),
                             "id4": (3, 4, 5, 3, 4, 5),
                             "id5": (5, 5),
                             "id6": (3, 4, 5, 3, 4, 5),
                             "id7": (2, 4, 5, 2, 4, 5),
                             "id8": (2, 5, 2, 5),
                             "id9": (2, 3, 5, 2, 3, 5)}
        node_open_leg_values = {"id1": [2, 3],
                                "id2": [3],
                                "id3": [2, 3],
                                "id4": [1, 2, 3],
                                "id5": [3],
                                "id6": [1, 2, 3],
                                "id7": [1, 2, 3],
                                "id8": [2, 3],
                                "id9": [1, 2, 3]}

        for node_id, node_tensor in self.tensortree.tensors.items():
            tensor = ptn.crandn(tensor_shape_dict[node_id])
            ref_tensor = deepcopy(node_tensor)
            tensor_legs = list(range(int(tensor.ndim / 2)))
            ref_tensor = np.tensordot(ref_tensor, tensor,
                                      axes=(node_open_leg_values[node_id], tensor_legs))

            self.tensortree.absorb_into_open_legs(node_id, tensor)
            found_tensor = self.tensortree.tensors[node_id]

            self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_tensor_contraction_leaf_only_child(self):
        self.tensortree.contract_nodes("id8", "id9")
        contracted_node, contracted_tensor = self.tensortree["id8contrid9"]

        # Test Node
        self.assertTrue(isinstance(contracted_node, ptn.Node))
        self.assertEqual(contracted_node.children, [])
        self.assertEqual(contracted_node.parent, "id1")

        # Test Tensor
        correct_shape = (3, 2, 5, 2, 3, 5)
        self.assertEqual(correct_shape, contracted_tensor.shape)

        # Test Connectivity
        self.assertTrue(self.tensortree.is_parent_of("id1", "id8contrid9"))

    def test_tensor_contraction_leaf_not_only_child(self):
        self.tensortree.contract_nodes("id5", "id7")
        contracted_id = "id5contrid7"
        contracted_node, contracted_tensor = self.tensortree[contracted_id]

        # Test Node
        self.assertTrue(isinstance(contracted_node, ptn.Node))
        self.assertEqual(contracted_node.children, ["id6"])
        self.assertEqual(contracted_node.parent, "id2")

        # Test Tensor
        correct_shape = (4, 2, 5, 2, 4, 5)
        self.assertEqual(correct_shape, contracted_tensor.shape)

        # Test Connectivity
        self.assertTrue(self.tensortree.is_parent_of("id2", contracted_id))
        self.assertTrue(self.tensortree.is_child_of("id6", contracted_id))

    def test_tensor_contraction_middle_node(self):
        self.tensortree.contract_nodes("id2", "id5")
        contracted_id = "id2contrid5"
        contracted_node, contracted_tensor = self.tensortree[contracted_id]

        # Test Node
        child_ids = ["id3", "id6", "id7"]
        self.assertTrue(isinstance(contracted_node, ptn.Node))
        self.assertEqual(contracted_node.children, child_ids)
        self.assertEqual(contracted_node.parent, "id1")

        # Test Tensor
        correct_shape = (2, 5, 2, 3, 3, 5)
        self.assertEqual(correct_shape, contracted_tensor.shape)

        # Test Connectivity
        self.assertTrue(self.tensortree.is_parent_of("id1", contracted_id))
        for child_id in child_ids:
            self.assertTrue(self.tensortree.is_child_of(child_id, contracted_id))

    def test_tensor_contraction_root(self):
        self.tensortree.contract_nodes("id1", "id2")
        contracted_id = "id1contrid2"
        contracted_node, contracted_tensor = self.tensortree[contracted_id]

        # Test Node
        child_ids = ["id8", "id3", "id5"]
        self.assertTrue(isinstance(contracted_node, ptn.Node))
        self.assertEqual(contracted_node.children, child_ids)
        self.assertTrue(contracted_node.is_root())

        # Test Tensor
        correct_shape = (3, 5, 4, 4, 5, 3)
        self.assertEqual(correct_shape, contracted_tensor.shape)

        # Test Connectivity
        for child_id in child_ids:
            self.assertTrue(self.tensortree.is_child_of(child_id, contracted_id))

    def test_tensor_split_leaf_q1parent_vs_r3open(self):
        q_legs = {"parent_leg": "id8", "child_legs": [], "open_legs": []}
        r_legs = {"parent_leg": None, "child_legs": [], "open_legs": [1, 2, 3]}
        self.tensortree.split_node_qr("id9", q_legs, r_legs,
                                      q_identifier="q9", r_identifier="r9")

        q_node, q_tensor = self.tensortree["q9"]
        r_node, r_tensor = self.tensortree["r9"]

        # Test Nodes
        self.assertEqual(10, len(self.tensortree.nodes))
        self.assertTrue("q9" in self.tensortree.nodes)
        self.assertTrue("r9" in self.tensortree.nodes)
        self.assertTrue(q_node.is_parent_of("r9"))
        self.assertTrue(r_node.is_child_of("q9"))

        # Test Tensors
        self.assertEqual((4, r_tensor.shape[0]), q_tensor.shape)
        self.assertEqual((q_tensor.shape[1], 2, 3, 5), r_tensor.shape)
        found_identity = np.tensordot(q_tensor, q_tensor.conj(), axes=(1, 1))
        self.assertTrue(np.allclose(np.eye(q_tensor.shape[0]), found_identity))

    def test_tensor_splitqr_leaf_r3open_vs_q1parent(self):
        r_legs = {"parent_leg": "id8", "child_legs": [], "open_legs": []}
        q_legs = {"parent_leg": None, "child_legs": [], "open_legs": [1, 2, 3]}
        self.tensortree.split_node_qr("id9", q_legs, r_legs,
                                      q_identifier="q9", r_identifier="r9")

        q_node, q_tensor = self.tensortree["q9"]
        r_node, r_tensor = self.tensortree["r9"]

        # Test Nodes
        self.assertEqual(10, len(self.tensortree.nodes))
        self.assertTrue("q9" in self.tensortree.nodes)
        self.assertTrue("r9" in self.tensortree.nodes)
        self.assertTrue(q_node.is_child_of("r9"))
        self.assertTrue(r_node.is_parent_of("q9"))

        # Test Tensors
        self.assertEqual((4, q_tensor.shape[0]), r_tensor.shape)
        self.assertEqual((r_tensor.shape[1], 2, 3, 5), q_tensor.shape)
        found_identity = np.tensordot(q_tensor, q_tensor.conj(), axes=([1, 2, 3], [1, 2, 3]))
        self.assertTrue(np.allclose(np.eye(q_tensor.shape[0]), found_identity))

    def test_tensor_splitqr_node_q1parent1open_vs_r1child1open(self):
        q_legs = {"parent_leg": "id1", "child_legs": [], "open_legs": [2]}
        r_legs = {"parent_leg": None, "child_legs": ["id9"], "open_legs": [3]}
        self.tensortree.split_node_qr("id8", q_legs, r_legs,
                                      q_identifier="q8", r_identifier="r8")

        q_node, q_tensor = self.tensortree["q8"]
        r_node, r_tensor = self.tensortree["r8"]

        # Test Nodes
        self.assertEqual(10, len(self.tensortree.nodes))
        self.assertTrue("q8" in self.tensortree.nodes)
        self.assertTrue("r8" in self.tensortree.nodes)
        self.assertTrue(q_node.is_parent_of("r8"))
        self.assertTrue(q_node.is_child_of("id1"))
        self.assertTrue(self.tensortree.nodes["id1"].is_parent_of("q8"))
        self.assertTrue(r_node.is_child_of("q8"))
        self.assertTrue(r_node.is_parent_of("id9"))
        self.assertTrue(self.tensortree.nodes["id9"].is_child_of("r8"))

        # Test Tensors
        self.assertEqual((3, r_tensor.shape[0], 2), q_tensor.shape)
        self.assertEqual((q_tensor.shape[1], 4, 5), r_tensor.shape)
        found_identity = np.tensordot(q_tensor, q_tensor.conj(), axes=([0, 2], [0, 2]))
        self.assertTrue(np.allclose(np.eye(q_tensor.shape[1]), found_identity))

    def test_tensor_splitqr_node_q1parent1child_vs_r1child1open(self):
        q_legs = {"parent_leg": "id2", "child_legs": ["id6"], "open_legs": []}
        r_legs = {"parent_leg": None, "child_legs": ["id7"], "open_legs": [3]}
        self.tensortree.split_node_qr("id5", q_legs, r_legs,
                                      q_identifier="q5", r_identifier="r5")

        q_node, q_tensor = self.tensortree["q5"]
        r_node, r_tensor = self.tensortree["r5"]

        # Test Nodes
        self.assertEqual(10, len(self.tensortree.nodes))
        self.assertTrue("q5" in self.tensortree.nodes)
        self.assertTrue("r5" in self.tensortree.nodes)
        self.assertTrue(q_node.is_parent_of("r5"))
        self.assertTrue(q_node.is_child_of("id2"))
        self.assertTrue(self.tensortree.nodes["id2"].is_parent_of("q5"))
        self.assertTrue(q_node.is_parent_of("id6"))
        self.assertTrue(self.tensortree.nodes["id6"].is_child_of("q5"))
        self.assertTrue(r_node.is_child_of("q5"))
        self.assertTrue(r_node.is_parent_of("id7"))
        self.assertTrue(self.tensortree.nodes["id7"].is_child_of("r5"))

        # Test Tensors
        self.assertEqual((4, 2, r_tensor.shape[0]), q_tensor.shape)
        self.assertEqual((q_tensor.shape[2], 3, 5), r_tensor.shape)
        found_identity = np.tensordot(q_tensor, q_tensor.conj(), axes=([0, 1], [0, 1]))
        self.assertTrue(np.allclose(np.eye(q_tensor.shape[2]), found_identity))

    def test_tensor_splitqr_root_q1child1open_vs_r1child1open(self):
        q_legs = {"parent_leg": None, "child_legs": ["id2"], "open_legs": [2]}
        r_legs = {"parent_leg": None, "child_legs": ["id8"], "open_legs": [3]}
        self.tensortree.split_node_qr("id1", q_legs, r_legs,
                                      q_identifier="q1", r_identifier="r1")

        q_node, q_tensor = self.tensortree["q1"]
        r_node, r_tensor = self.tensortree["r1"]

        # Test Nodes
        self.assertEqual(10, len(self.tensortree.nodes))
        self.assertTrue("q1" in self.tensortree.nodes)
        self.assertTrue("r1" in self.tensortree.nodes)
        self.assertTrue(q_node.is_parent_of("r1"))
        self.assertTrue(q_node.is_root())
        self.assertTrue(self.tensortree.root_id == q_node.identifier)
        self.assertTrue(q_node.is_parent_of("id2"))
        self.assertTrue(self.tensortree.nodes["id2"].is_child_of("q1"))
        self.assertTrue(r_node.is_child_of("q1"))
        self.assertTrue(r_node.is_parent_of("id8"))
        self.assertTrue(self.tensortree.nodes["id8"].is_child_of("r1"))

        # Test Tensors
        self.assertEqual((2, r_tensor.shape[0], 4), q_tensor.shape)
        self.assertEqual((q_tensor.shape[1], 3, 5), r_tensor.shape)
        found_identity = np.tensordot(q_tensor, q_tensor.conj(), axes=([0, 2], [0, 2]))
        self.assertTrue(np.allclose(np.eye(q_tensor.shape[1]), found_identity))

    def test_tensor_splitsvd_leaf_q1parent_vs_r3open(self):
        u_legs = {"parent_leg": "id8", "child_legs": [], "open_legs": []}
        v_legs = {"parent_leg": None, "child_legs": [], "open_legs": [1, 2, 3]}
        self.tensortree.split_node_svd("id9", u_legs, v_legs,
                                       u_identifier="q9", v_identifier="r9")

        u_node, u_tensor = self.tensortree["q9"]
        v_node, v_tensor = self.tensortree["r9"]

        # Test Nodes
        self.assertEqual(10, len(self.tensortree.nodes))
        self.assertTrue("q9" in self.tensortree.nodes)
        self.assertTrue("r9" in self.tensortree.nodes)
        self.assertTrue(u_node.is_parent_of("r9"))
        self.assertTrue(v_node.is_child_of("q9"))

        # Test Tensors
        self.assertEqual((4, v_tensor.shape[0]), u_tensor.shape)
        self.assertEqual((u_tensor.shape[1], 2, 3, 5), v_tensor.shape)
        found_identity = np.tensordot(u_tensor, u_tensor.conj(), axes=(1, 1))
        self.assertTrue(np.allclose(np.eye(u_tensor.shape[0]), found_identity))

    def test_tensor_splitsvd_leaf_r3open_vs_q1parent(self):
        v_legs = {"parent_leg": "id8", "child_legs": [], "open_legs": []}
        u_legs = {"parent_leg": None, "child_legs": [], "open_legs": [1, 2, 3]}
        self.tensortree.split_node_svd("id9", u_legs, v_legs,
                                       u_identifier="q9", v_identifier="r9")

        u_node, u_tensor = self.tensortree["q9"]
        v_node, v_tensor = self.tensortree["r9"]

        # Test Nodes
        self.assertEqual(10, len(self.tensortree.nodes))
        self.assertTrue("q9" in self.tensortree.nodes)
        self.assertTrue("r9" in self.tensortree.nodes)
        self.assertTrue(u_node.is_child_of("r9"))
        self.assertTrue(v_node.is_parent_of("q9"))

        # Test Tensors
        self.assertEqual((4, u_tensor.shape[0]), v_tensor.shape)
        self.assertEqual((v_tensor.shape[1], 2, 3, 5), u_tensor.shape)
        found_identity = np.tensordot(u_tensor, u_tensor.conj(), axes=([1, 2, 3], [1, 2, 3]))
        self.assertTrue(np.allclose(np.eye(u_tensor.shape[0]), found_identity))

    def test_tensor_splitsvd_node_q1parent1open_vs_r1child1open(self):
        u_legs = {"parent_leg": "id1", "child_legs": [], "open_legs": [2]}
        v_legs = {"parent_leg": None, "child_legs": ["id9"], "open_legs": [3]}
        self.tensortree.split_node_svd("id8", u_legs, v_legs,
                                       u_identifier="q8", v_identifier="r8")

        u_node, u_tensor = self.tensortree["q8"]
        v_node, v_tensor = self.tensortree["r8"]

        # Test Nodes
        self.assertEqual(10, len(self.tensortree.nodes))
        self.assertTrue("q8" in self.tensortree.nodes)
        self.assertTrue("r8" in self.tensortree.nodes)
        self.assertTrue(u_node.is_parent_of("r8"))
        self.assertTrue(u_node.is_child_of("id1"))
        self.assertTrue(self.tensortree.nodes["id1"].is_parent_of("q8"))
        self.assertTrue(v_node.is_child_of("q8"))
        self.assertTrue(v_node.is_parent_of("id9"))
        self.assertTrue(self.tensortree.nodes["id9"].is_child_of("r8"))

        # Test Tensors
        self.assertEqual((3, v_tensor.shape[0], 2), u_tensor.shape)
        self.assertEqual((u_tensor.shape[1], 4, 5), v_tensor.shape)
        found_identity = np.tensordot(u_tensor, u_tensor.conj(), axes=([0, 2], [0, 2]))
        self.assertTrue(np.allclose(np.eye(u_tensor.shape[1]), found_identity))

    def test_tensor_splitsvd_node_q1parent1child_vs_r1child1open(self):
        u_legs = {"parent_leg": "id2", "child_legs": ["id6"], "open_legs": []}
        v_legs = {"parent_leg": None, "child_legs": ["id7"], "open_legs": [3]}
        self.tensortree.split_node_svd("id5", u_legs, v_legs,
                                       u_identifier="q5", v_identifier="r5")

        u_node, u_tensor = self.tensortree["q5"]
        v_node, v_tensor = self.tensortree["r5"]

        # Test Nodes
        self.assertEqual(10, len(self.tensortree.nodes))
        self.assertTrue("q5" in self.tensortree.nodes)
        self.assertTrue("r5" in self.tensortree.nodes)
        self.assertTrue(u_node.is_parent_of("r5"))
        self.assertTrue(u_node.is_child_of("id2"))
        self.assertTrue(self.tensortree.nodes["id2"].is_parent_of("q5"))
        self.assertTrue(u_node.is_parent_of("id6"))
        self.assertTrue(self.tensortree.nodes["id6"].is_child_of("q5"))
        self.assertTrue(v_node.is_child_of("q5"))
        self.assertTrue(v_node.is_parent_of("id7"))
        self.assertTrue(self.tensortree.nodes["id7"].is_child_of("r5"))

        # Test Tensors
        self.assertEqual((4, 2, v_tensor.shape[0]), u_tensor.shape)
        self.assertEqual((u_tensor.shape[2], 3, 5), v_tensor.shape)
        found_identity = np.tensordot(u_tensor, u_tensor.conj(), axes=([0, 1], [0, 1]))
        self.assertTrue(np.allclose(np.eye(u_tensor.shape[2]), found_identity))

    def test_tensor_splitsvd_root_q1child1open_vs_r1child1open(self):
        u_legs = {"parent_leg": None, "child_legs": ["id2"], "open_legs": [2]}
        v_legs = {"parent_leg": None, "child_legs": ["id8"], "open_legs": [3]}
        self.tensortree.split_node_svd("id1", u_legs, v_legs,
                                       u_identifier="q1", v_identifier="r1")

        u_node, u_tensor = self.tensortree["q1"]
        v_node, v_tensor = self.tensortree["r1"]

        # Test Nodes
        self.assertEqual(10, len(self.tensortree.nodes))
        self.assertTrue("q1" in self.tensortree.nodes)
        self.assertTrue("r1" in self.tensortree.nodes)
        self.assertTrue(u_node.is_parent_of("r1"))
        self.assertTrue(u_node.is_root())
        self.assertTrue(self.tensortree.root_id == u_node.identifier)
        self.assertTrue(u_node.is_parent_of("id2"))
        self.assertTrue(self.tensortree.nodes["id2"].is_child_of("q1"))
        self.assertTrue(v_node.is_child_of("q1"))
        self.assertTrue(v_node.is_parent_of("id8"))
        self.assertTrue(self.tensortree.nodes["id8"].is_child_of("r1"))

        # Test Tensors
        self.assertEqual((2, v_tensor.shape[0], 4), u_tensor.shape)
        self.assertEqual((u_tensor.shape[1], 3, 5), v_tensor.shape)
        found_identity = np.tensordot(u_tensor, u_tensor.conj(), axes=([0, 2], [0, 2]))
        self.assertTrue(np.allclose(np.eye(u_tensor.shape[1]), found_identity))

    # TODO: Reactivate later
    # def test_apply_hamiltonian(self):
    #     ttns = ptn.TreeTensorNetwork()
    #     node1, tensor1 = ptn.random_tensor_node((2, 3), identifier="site1")
    #     node2, tensor2 = ptn.random_tensor_node((2, 4), identifier="site2")

    #     ttns.add_root(node1, tensor1)
    #     ttns.add_child_to_parent(node2, tensor2, 0, "site1", 0)

    #     full_state = ttns.completely_contract_tree(to_copy=True).tensors["site1_contr_site2"]

    #     term1 = {"site1": "11", "site2": "21"}
    #     term2 = {"site1": "12", "site2": "22"}
    #     conversion_dict = {"11": ptn.crandn((3, 3)), "12": ptn.crandn((3, 3)),
    #                        "21": ptn.crandn((4, 4)), "22": ptn.crandn((4, 4))}

    #     hamiltonian = ptn.Hamiltonian(terms=[term1, term2],
    #                                   conversion_dictionary=conversion_dict)

    #     ttns.apply_hamiltonian(hamiltonian, conversion_dict)
    #     test_output = ttns.completely_contract_tree(to_copy=True).tensors["site1_contr_site2"]

    #     full_tensor = hamiltonian.to_tensor(ttns)
    #     ttno_output = np.tensordot(full_state, full_tensor, axes=([0, 1], [2, 3]))
    #     np.allclose(test_output, ttno_output)


if __name__ == "__main__":
    unittest.main()
