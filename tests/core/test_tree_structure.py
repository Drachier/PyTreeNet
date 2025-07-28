"""
This module implements unittests for the tree structure class.
"""

import unittest

from copy import deepcopy
from itertools import combinations

import pytreenet as ptn
from pytreenet.random import crandn
from pytreenet.special_ttn.star import StarTreeTensorState
from pytreenet.special_ttn.binary import generate_binary_ttns

class TestTreeStructureInit(unittest.TestCase):
    def test_init(self):
        ts = ptn.TreeStructure()
        self.assertEqual(None, ts.root_id)
        self.assertEqual(0, len(ts.nodes))

class TestTreeStructureBuildingMethods(unittest.TestCase):
    def setUp(self) -> None:
        # We need to create some nodes to add to the tree structure
        self.identifiers = ["node" + str(i) for i in range(9)]
        self.nodes =  [ptn.GraphNode(identifier=self.identifiers[i])
                 for i in range(len(self.identifiers))]
        self.ts = ptn.TreeStructure()

    def test_add_root(self):
        self.ts.add_root(self.nodes[0])
        self.assertEqual(self.identifiers[0], self.ts.root_id)
        self.assertEqual(1, len(self.ts.nodes))

    def test_add_child_to_parent(self):
        self.ts.add_root(self.nodes[0])
        self.ts.add_child_to_parent(self.nodes[1], self.identifiers[0])
        self.ts.add_child_to_parent(self.nodes[2], self.identifiers[1])
        self.ts.add_child_to_parent(self.nodes[3], self.identifiers[2])
        self.ts.add_child_to_parent(self.nodes[4], self.identifiers[1])
        self.ts.add_child_to_parent(self.nodes[5], self.identifiers[4])
        self.ts.add_child_to_parent(self.nodes[6], self.identifiers[4])
        self.ts.add_child_to_parent(self.nodes[7], self.identifiers[0])
        self.ts.add_child_to_parent(self.nodes[8], self.identifiers[7])

        self.assertTrue(self.nodes[0].is_parent_of(self.identifiers[1]))
        self.assertTrue(self.nodes[0].is_parent_of(self.identifiers[7]))
        self.assertTrue(self.nodes[0].is_root())
        self.assertTrue(self.nodes[1].is_parent_of(self.identifiers[2]))
        self.assertTrue(self.nodes[1].is_parent_of(self.identifiers[4]))
        self.assertTrue(self.nodes[1].is_child_of(self.identifiers[0]))
        self.assertTrue(self.nodes[2].is_parent_of(self.identifiers[3]))
        self.assertTrue(self.nodes[2].is_child_of(self.identifiers[1]))
        self.assertTrue(self.nodes[3].is_leaf())
        self.assertTrue(self.nodes[4].is_parent_of(self.identifiers[5]))
        self.assertTrue(self.nodes[4].is_parent_of(self.identifiers[6]))
        self.assertTrue(self.nodes[4].is_child_of(self.identifiers[1]))
        self.assertTrue(self.nodes[5].is_child_of(self.identifiers[4]))
        self.assertTrue(self.nodes[5].is_leaf())
        self.assertTrue(self.nodes[6].is_child_of(self.identifiers[4]))
        self.assertTrue(self.nodes[6].is_leaf())
        self.assertTrue(self.nodes[7].is_parent_of(self.identifiers[8]))
        self.assertTrue(self.nodes[7].is_child_of(self.identifiers[0]))
        self.assertTrue(self.nodes[8].is_child_of(self.identifiers[7]))
        self.assertTrue(self.nodes[8].is_leaf())

    def test_add_parent_to_root(self):
        self.ts.add_root(self.nodes[0])
        self.ts.add_parent_to_root(self.nodes[1])
        self.assertTrue(self.nodes[0].is_leaf())
        self.assertTrue(self.nodes[0].is_child_of(self.identifiers[1]))
        self.assertTrue(self.nodes[1].is_root())
        self.assertTrue(self.nodes[1].is_parent_of(self.identifiers[0]))

class TestTreeStructureMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.identifiers = ["node" + str(i) for i in range(9)]
        self.nodes =  [ptn.GraphNode(identifier=self.identifiers[i])
                 for i in range(len(self.identifiers))]
        self.ts = ptn.TreeStructure()
        self.ts.add_root(self.nodes[0])
        self.ts.add_child_to_parent(self.nodes[1], self.identifiers[0])
        self.ts.add_child_to_parent(self.nodes[2], self.identifiers[1])
        self.ts.add_child_to_parent(self.nodes[3], self.identifiers[2])
        self.ts.add_child_to_parent(self.nodes[4], self.identifiers[1])
        self.ts.add_child_to_parent(self.nodes[5], self.identifiers[4])
        self.ts.add_child_to_parent(self.nodes[6], self.identifiers[4])
        self.ts.add_child_to_parent(self.nodes[7], self.identifiers[0])
        self.ts.add_child_to_parent(self.nodes[8], self.identifiers[7])

    def test_nearest_neighbours(self):
        correct_neighbours = [(self.identifiers[0],self.identifiers[1]),
                              (self.identifiers[0],self.identifiers[7]),
                              (self.identifiers[1],self.identifiers[2]),
                              (self.identifiers[1],self.identifiers[4]),
                              (self.identifiers[2],self.identifiers[3]),
                              (self.identifiers[4],self.identifiers[5]),
                              (self.identifiers[4],self.identifiers[6]),
                              (self.identifiers[7],self.identifiers[8])]
        correct_neighbours = {frozenset(pair) for pair in correct_neighbours}
        found = {frozenset(pair) for pair in self.ts.nearest_neighbours()}
        self.assertEqual(correct_neighbours, found)

    def test_get_leaves(self):
        correct_leaves = {self.identifiers[3],
                          self.identifiers[5],
                          self.identifiers[6], 
                          self.identifiers[8]}
        self.assertEqual(correct_leaves, set(self.ts.get_leaves()))

    def test_distance_to_node_root(self):
        correct_distances = {self.identifiers[0]: 0,
                             self.identifiers[1]: 1,
                             self.identifiers[2]: 2,
                             self.identifiers[3]: 3,
                             self.identifiers[4]: 2,
                             self.identifiers[5]: 3,
                             self.identifiers[6]: 3,
                             self.identifiers[7]: 1,
                             self.identifiers[8]: 2}
        found_distances = self.ts.distance_to_node(self.identifiers[0])
        self.assertEqual(correct_distances, found_distances)

    def test_determine_parentage(self):
        for node in self.nodes:
            for child_id in node.children:
                identifier = node.identifier
                correct = (identifier, child_id)
                found1 = self.ts.determine_parentage(identifier, child_id)
                self.assertEqual(correct, found1)
                found2 = self.ts.determine_parentage(child_id, identifier)
                self.assertEqual(correct, found2)

    def test_replace_node_in_neighbours(self):
        new_node = ptn.GraphNode(identifier="new")
        node1 = self.nodes[1]
        new_node.children = deepcopy(node1.children)
        new_node.parent = node1.parent
        self.ts._nodes["new"] = new_node
        self.ts.replace_node_in_neighbours("new", self.identifiers[1])
        self.assertTrue(self.nodes[0].is_parent_of("new"))
        self.assertTrue(self.nodes[2].is_child_of("new"))
        self.assertTrue(self.nodes[4].is_child_of("new"))
        self.assertFalse(self.identifiers[1] in self.ts)
        self.assertFalse(self.nodes[0].is_parent_of(self.identifiers[1]))
        self.assertFalse(self.nodes[2].is_child_of(self.identifiers[1]))
        self.assertFalse(self.nodes[4].is_child_of(self.identifiers[1]))

    def test_replace_node_in_neighbours_root(self):
        new_node = ptn.GraphNode(identifier="new")
        node0 = self.nodes[0]
        new_node.children = deepcopy(node0.children)
        new_node.parent = node0.parent
        self.ts._nodes["new"] = new_node
        self.ts.replace_node_in_neighbours("new", self.identifiers[0])
        self.assertTrue(self.nodes[1].is_child_of("new"))
        self.assertTrue(self.nodes[7].is_child_of("new"))
        self.assertEqual("new", self.ts.root_id)
        self.assertFalse(self.identifiers[0] in self.ts)
        self.assertFalse(self.nodes[1].is_child_of(self.identifiers[1]))
        self.assertFalse(self.nodes[7].is_child_of(self.identifiers[1]))

    def test_replace_node_in_some_neighbours_parent_and_children(self):
        new_node = ptn.GraphNode(identifier="new")
        new_node.children = [self.identifiers[5]]
        new_node.parent = self.identifiers[1]
        self.ts._nodes["new"] = new_node
        self.ts.replace_node_in_some_neighbours("new",
                                                self.identifiers[4],
                                                [self.identifiers[1],self.identifiers[5]])
        self.assertTrue(self.nodes[1].is_parent_of("new"))
        self.assertTrue(self.nodes[5].is_child_of("new"))
        self.assertTrue(self.nodes[6].is_child_of(self.identifiers[4]))
        self.assertFalse(self.nodes[1].is_parent_of(self.identifiers[4]))
        self.assertFalse(self.nodes[5].is_child_of(self.identifiers[4]))
        self.assertFalse(self.nodes[6].is_child_of("new"))

    def test_replace_node_in_some_neighbours_leaf(self):
        new_node = ptn.GraphNode(identifier="new")
        new_node.parent = self.identifiers[2]
        self.ts._nodes["new"] = new_node
        self.ts.replace_node_in_some_neighbours("new",
                                                self.identifiers[3],
                                                [self.identifiers[2]])
        self.assertTrue(self.nodes[2].is_parent_of("new"))
        self.assertFalse(self.nodes[2].is_parent_of(self.identifiers[3]))

    def test_replace_node_in_some_neighbours_root(self):
        new_node = ptn.GraphNode(identifier="new")
        new_node.parent = self.identifiers[2]
        self.ts._nodes["new"] = new_node
        self.ts.replace_node_in_some_neighbours("new",
                                                self.identifiers[3],
                                                [self.identifiers[2]])
        self.assertTrue(self.nodes[2].is_parent_of("new"))
        self.assertFalse(self.nodes[2].is_parent_of(self.identifiers[3]))

    def test_find_path_to_root(self):
        correct_paths = {self.identifiers[0]: [self.identifiers[0]],
                         self.identifiers[1]: [self.identifiers[1], self.identifiers[0]],
                         self.identifiers[2]: [self.identifiers[2], self.identifiers[1], self.identifiers[0]],
                         self.identifiers[3]: [self.identifiers[3], self.identifiers[2], self.identifiers[1], self.identifiers[0]],
                         self.identifiers[4]: [self.identifiers[4], self.identifiers[1], self.identifiers[0]],
                         self.identifiers[5]: [self.identifiers[5], self.identifiers[4], self.identifiers[1], self.identifiers[0]],
                         self.identifiers[6]: [self.identifiers[6], self.identifiers[4], self.identifiers[1], self.identifiers[0]],
                         self.identifiers[7]: [self.identifiers[7], self.identifiers[0]],
                         self.identifiers[8]: [self.identifiers[8], self.identifiers[7], self.identifiers[0]]}
        for start_id, correct_path in correct_paths.items():
            found_path = self.ts.find_path_to_root(start_id)
            self.assertEqual(correct_path, found_path)

    def test_path_from_self_to_self(self):
        for identifier in self.ts.nodes:
            correct_path = [identifier]
            found_path = self.ts.path_from_to(identifier, identifier)
            self.assertEqual(correct_path, found_path)

    def test_path_from_to_root(self):
        # These paths should be equal to the above paths.
        correct_paths = {self.identifiers[0]: [self.identifiers[0]],
                         self.identifiers[1]: [self.identifiers[1], self.identifiers[0]],
                         self.identifiers[2]: [self.identifiers[2], self.identifiers[1], self.identifiers[0]],
                         self.identifiers[3]: [self.identifiers[3], self.identifiers[2], self.identifiers[1], self.identifiers[0]],
                         self.identifiers[4]: [self.identifiers[4], self.identifiers[1], self.identifiers[0]],
                         self.identifiers[5]: [self.identifiers[5], self.identifiers[4], self.identifiers[1], self.identifiers[0]],
                         self.identifiers[6]: [self.identifiers[6], self.identifiers[4], self.identifiers[1], self.identifiers[0]],
                         self.identifiers[7]: [self.identifiers[7], self.identifiers[0]],
                         self.identifiers[8]: [self.identifiers[8], self.identifiers[7], self.identifiers[0]]}
        for start_id, correct_path in correct_paths.items():
            found_path = self.ts.path_from_to(start_id, self.ts.root_id)
            self.assertEqual(correct_path, found_path)
    
    def test_path_from_leaf_to_leaf(self):
        correct_paths = {(self.identifiers[3], self.identifiers[5]):
                         [self.identifiers[3], self.identifiers[2], self.identifiers[1], self.identifiers[4], self.identifiers[5]],
                         (self.identifiers[6], self.identifiers[3]):
                         [self.identifiers[6], self.identifiers[4], self.identifiers[1], self.identifiers[2], self.identifiers[3]],
                         (self.identifiers[5], self.identifiers[6]):
                         [self.identifiers[5], self.identifiers[4], self.identifiers[6]],
                         (self.identifiers[3], self.identifiers[8]):
                         [self.identifiers[3], self.identifiers[2], self.identifiers[1], self.identifiers[0], self.identifiers[7], self.identifiers[8]],
                         (self.identifiers[8], self.identifiers[5]):
                         [self.identifiers[8], self.identifiers[7], self.identifiers[0], self.identifiers[1], self.identifiers[4], self.identifiers[5]]}
        for identifiers, correct_path in correct_paths.items():
            found_path = self.ts.path_from_to(identifiers[0], identifiers[1])
            self.assertEqual(correct_path, found_path)

    def test_path_from_to(self):
        correct_paths = {(self.identifiers[3], self.identifiers[7]):
                         [self.identifiers[3], self.identifiers[2], self.identifiers[1], self.identifiers[0], self.identifiers[7]],
                         (self.identifiers[2], self.identifiers[4]):
                         [self.identifiers[2], self.identifiers[1], self.identifiers[4]],
                         (self.identifiers[0], self.identifiers[6]):
                         [self.identifiers[0], self.identifiers[1], self.identifiers[4], self.identifiers[6]],
                         (self.identifiers[5], self.identifiers[1]):
                         [self.identifiers[5], self.identifiers[4], self.identifiers[1]]}
        for identifiers, correct_path in correct_paths.items():
            found_path = self.ts.path_from_to(identifiers[0], identifiers[1])
            self.assertEqual(correct_path, found_path)

    def test_linearise(self):
        found_list = self.ts.linearise()
        correct_list = ["node3", "node2", "node5", "node6", "node4",
                        "node1", "node8", "node7", "node0"]
        self.assertEqual(correct_list,found_list)

    def test_find_pairs_of_distance_0(self):
        """
        Test that all pairs of nodes that are at distance 0 from each other
        are found correctly.
        """
        pairs = self.ts.find_pairs_of_distance(0)
        correct_pairs = {(self.identifiers[i], self.identifiers[i])
                         for i in range(len(self.identifiers))}
        correct_pairs = {frozenset(pair) for pair in correct_pairs}
        self.assertEqual(set(pairs), correct_pairs)

    def test_find_pairs_of_distance_1(self):
        """
        Test that all pairs of nodes that are at distance 1 from each other
        are found correctly.
        """
        pairs = self.ts.find_pairs_of_distance(1)
        correct_pairs = {(self.identifiers[0], self.identifiers[1]),
                         (self.identifiers[0], self.identifiers[7]),
                         (self.identifiers[1], self.identifiers[2]),
                         (self.identifiers[1], self.identifiers[4]),
                         (self.identifiers[2], self.identifiers[3]),
                         (self.identifiers[4], self.identifiers[5]),
                         (self.identifiers[4], self.identifiers[6]),
                         (self.identifiers[7], self.identifiers[8])}
        correct_pairs = {frozenset(pair) for pair in correct_pairs}
        self.assertEqual(pairs, correct_pairs)

    def test_find_pairs_of_distance_2(self):
        """
        Test that all pairs of nodes that are at distance 2 from each other
        are found correctly.
        """
        pairs = self.ts.find_pairs_of_distance(2)
        correct_pairs = {(self.identifiers[0], self.identifiers[2]),
                         (self.identifiers[0], self.identifiers[4]),
                         (self.identifiers[0], self.identifiers[8]),
                         (self.identifiers[1], self.identifiers[3]),
                         (self.identifiers[1], self.identifiers[5]),
                         (self.identifiers[1], self.identifiers[6]),
                         (self.identifiers[1], self.identifiers[7]),
                         (self.identifiers[6], self.identifiers[5]),
                         (self.identifiers[2], self.identifiers[4])}
        correct_pairs = {frozenset(pair) for pair in correct_pairs}
        self.assertEqual(set(pairs), correct_pairs)

    def test_find_pairs_of_distance_3(self):
        """
        Test that all pairs of nodes that are at distance 3 from each other
        are found correctly.
        """
        pairs = self.ts.find_pairs_of_distance(3)
        correct_pairs = {(self.identifiers[8], self.identifiers[1]),
                         (self.identifiers[2], self.identifiers[7]),
                         (self.identifiers[4], self.identifiers[7]),
                         (self.identifiers[0], self.identifiers[3]),
                         (self.identifiers[0], self.identifiers[5]),
                         (self.identifiers[0], self.identifiers[6]),
                         (self.identifiers[3], self.identifiers[4]),
                         (self.identifiers[2], self.identifiers[5]),
                         (self.identifiers[2], self.identifiers[6])}
        correct_pairs = {frozenset(pair) for pair in correct_pairs}
        self.assertEqual(set(pairs), correct_pairs)

    def test_find_pairs_of_distance_4(self):
        """
        Test that all pairs of nodes that are at distance 4 from each other
        are found correctly.
        """
        pairs = self.ts.find_pairs_of_distance(4)
        correct_pairs = {(self.identifiers[8], self.identifiers[2]),
                         (self.identifiers[8], self.identifiers[4]),
                         (self.identifiers[7], self.identifiers[3]),
                         (self.identifiers[7], self.identifiers[5]),
                         (self.identifiers[7], self.identifiers[6]),
                         (self.identifiers[3], self.identifiers[5]),
                         (self.identifiers[3], self.identifiers[6])}
        correct_pairs = {frozenset(pair) for pair in correct_pairs}
        self.assertEqual(set(pairs), correct_pairs)

    def test_find_pairs_of_distance_5(self):
        """
        Test that all pairs of nodes that are at distance 5 from each other
        are found correctly.
        """
        pairs = self.ts.find_pairs_of_distance(5)
        correct_pairs = {(self.identifiers[8], self.identifiers[3]),
                         (self.identifiers[8], self.identifiers[5]),
                         (self.identifiers[8], self.identifiers[6])}
        correct_pairs = {frozenset(pair) for pair in correct_pairs}
        self.assertEqual(set(pairs), correct_pairs)

    def test_find_pairs_of_distance_none(self):
        """
        Test that for a too large distance, no pairs are found.
        """
        pairs = self.ts.find_pairs_of_distance(6)
        self.assertEqual(set(pairs), set())

class TestDistancePairsWithOpenLegs(unittest.TestCase):
    """
    Tests the pair finding while considering open legs.
    This is a separate test case, as it requires `Nodes` and not `GraphNodes`.
    """

    def setUp(self) -> None:
        self.binary = generate_binary_ttns(6,2,crandn(2,3))
        center_tensor = crandn(2,2,2)
        chain_tensors = [[crandn(2,2,3),crandn(2,3)]
                         for _ in range(3)]
        self.t_ttn = StarTreeTensorState.from_tensor_lists(center_tensor,
                                                           chain_tensors)

    def test_find_pairs_of_distance_0_binary(self):
        """
        Test the finding of pairs of distance 0 while considering open legs
        for a binary tree.
        """
        found = self.binary.find_pairs_of_distance(0,
                                                   consider_open=True)
        correct = {frozenset([f"site{i}"]) for i in range(6)}
        self.assertEqual(correct,found)

    def test_find_pairs_of_distance_1_binary(self):
        """
        Test the finding of pairs of distance 1 while considering only open
        legs for a binary tree. This should be an all to all pairing of the
        physical nodes.
        """
        found = self.binary.find_pairs_of_distance(1,
                                                   consider_open=True)
        node_ids = [f"site{i}" for i in range(6)]
        correct = combinations(node_ids, r=2)
        correct = {frozenset(nodes) for nodes in correct}
        self.assertEqual(correct,found)

    def test_find_pairs_of_distance_2_binary(self):
        """
        Test the finding of pairs of distance 2 while considering only open
        legs in a binary tree. This should lead to no pairs at all.
        """
        found = self.binary.find_pairs_of_distance(2,
                                                   consider_open=True)
        correct = set()
        self.assertEqual(correct,found)

    def test_find_pairs_of_distance_0_tstructure(self):
        """
        Test the finding of pairs of distance 0 while considering only open
        legs in a T-shaped tree. This should find all individual nodes
        except for the center node.
        """
        found = self.t_ttn.find_pairs_of_distance(0,
                                                  consider_open=True)
        correct = {frozenset([node_id]) for node_id in self.t_ttn.nodes.keys()
                   if node_id != self.t_ttn.central_node_identifier}
        self.assertEqual(correct,found)

    def test_find_pairs_of_distance_1_tstructure(self):
        """
        Test the finding of pairs of distance 1 while considering only open
        legs in a T-shaped tree.
        """
        found = self.t_ttn.find_pairs_of_distance(1,
                                                  consider_open=True)
        correct = {("node0_0","node0_1"),
                   ("node0_0","node1_0"),
                   ("node0_0","node2_0"),
                   ("node1_0","node2_0"),
                   ("node1_0","node1_1"),
                   ("node2_0","node2_1")}
        correct = {frozenset(pair) for pair in correct}
        self.assertEqual(correct, found)

    def test_find_pairs_of_distance_2_tstructure(self):
        """
        Test the finding of pairs with distance 2 while considering only open
        legs for a T-shaped tree.
        """
        found = self.t_ttn.find_pairs_of_distance(2,
                                                  consider_open=True)
        correct = {("node0_1","node1_0"),
                   ("node0_1","node2_0"),
                   ("node1_1","node0_0"),
                   ("node1_1","node2_0"),
                   ("node2_1","node0_0"),
                   ("node2_1","node1_0"),
                   }
        correct = {frozenset(pair) for pair in correct}
        self.assertEqual(correct, found)

    def test_find_pairs_of_distance_3_tstructure(self):
        """
        Test the finding of node pairs with distance 3 while considering
        only open legs on a T-shaped tree.
        """
        found = self.t_ttn.find_pairs_of_distance(3,
                                                  consider_open=True)
        correct = {("node0_1","node1_1"),
                   ("node0_1","node2_1"),
                   ("node1_1","node2_1"),
                   }
        correct = {frozenset(pair) for pair in correct}
        self.assertEqual(correct, found)

    def test_find_pairs_of_distance_4_tstructure(self):
        """
        Test the finding of node pairs with distance 4 while considering
        only open legs on a T-shaped tree. There shouldn't be any.
        """
        found = self.t_ttn.find_pairs_of_distance(4,
                                                  consider_open=True)
        correct = set()
        self.assertEqual(correct, found)


if __name__ == "__main__":
    unittest.main()
