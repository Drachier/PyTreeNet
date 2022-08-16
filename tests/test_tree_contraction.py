import unittest

import numpy as np

import pytreenet as ptn

class TestTreeTensorNetworkBasics(unittest.TestCase):

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

    def test_construct_contracted_identifier(self):
        new_id1 = ptn.construct_contracted_identifier(self.tree_tensor_network,
                                            "id1", "id2")
        self.assertEqual(new_id1, "id1_contr_id2")

        new_id2 = ptn.construct_contracted_identifier(self.tree_tensor_network,
                                            "id1", "id2", "new")
        self.assertEqual(new_id2, "new")

    def test_find_connecting_legs(self):
        leg2to4, leg4to2 = ptn.find_connecting_legs(self.node2, self.node4)
        self.assertEqual(leg2to4, 2)
        self.assertEqual(leg4to2, 0)

    def test_combine_nodes(self):
        self.assertRaises(ptn.NoConnectionException ,
                          ptn.combine_nodes, self.tree_tensor_network,
                          "node1", "node3")

        nodes = self.tree_tensor_network.nodes

        ptn.combine_nodes(self.tree_tensor_network, "node2", "node4")
        idcontr24 = "node2_contr_node4"
        self.assertTrue(idcontr24 in nodes)
        self.assertEqual(len(nodes), 6)
        self.assertFalse("node2" in nodes)
        self.assertFalse("node4" in nodes)

        contracted_node = nodes[idcontr24]
        self.assertEqual(contracted_node.open_legs, [2])
        self.assertEqual(contracted_node.parent_leg, ["node1", 0])
        self.assertEqual(contracted_node.children_legs, {"node3":1})

        self.assertFalse(self.node2 == self.node4)
        self.assertEqual(nodes["node3"].parent_leg, [idcontr24,0])
        self.assertEqual(nodes["node1"].children_legs, {idcontr24: 0, "node5": 1})

        ref_tensor = np.tensordot(self.node2.tensor, self.node4.tensor, axes=(2,0))
        self.assertTrue(np.allclose(contracted_node.tensor, ref_tensor))

        ptn.combine_nodes(self.tree_tensor_network, "node1", idcontr24)
        idcontr124 = "node1_contr_node2_contr_node4"
        self.assertTrue(idcontr124 in nodes)
        self.assertEqual(self.tree_tensor_network.root_id, idcontr124)
        self.assertTrue(nodes[idcontr124].is_root())
        self.assertEqual(nodes[idcontr124].tensor.shape, (2,3,4))

        ptn.combine_nodes(self.tree_tensor_network, idcontr124, "node3")
        idcontr1243 = idcontr124 + "_contr_node3"
        self.assertTrue(idcontr1243 in nodes)
        self.assertEqual(nodes[idcontr1243].children_legs, {"node5": 0})
        self.assertEqual(nodes[idcontr1243].open_legs, [1])
        self.assertTrue(nodes[idcontr1243].is_root())

        ptn.combine_nodes(self.tree_tensor_network,
                          self.tree_tensor_network.root_id, "node5")
        ptn.combine_nodes(self.tree_tensor_network,
                          self.tree_tensor_network.root_id, "node6")
        ptn.combine_nodes(self.tree_tensor_network,
                          self.tree_tensor_network.root_id, "node7")

        id_all_contracted = self.tree_tensor_network.root_id
        self.assertEqual(len(nodes),1)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[id_all_contracted].children_legs, dict())
        self.assertEqual(nodes[id_all_contracted].open_legs, [0,1,2,3,4])

if __name__ == "__main__":
    unittest.main()