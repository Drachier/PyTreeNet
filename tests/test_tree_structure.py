import unittest

from copy import deepcopy

import pytreenet as ptn

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
        correct_neighbours = [(self.identifiers[0],self.identifiers[1]), (self.identifiers[0],self.identifiers[7]),
                              (self.identifiers[1],self.identifiers[2]), (self.identifiers[1],self.identifiers[4]),
                              (self.identifiers[2],self.identifiers[3]),
                              (self.identifiers[4],self.identifiers[5]), (self.identifiers[4],self.identifiers[6]),
                              (self.identifiers[7],self.identifiers[8])]
        self.assertEqual(set(correct_neighbours), set(self.ts.nearest_neighbours()))

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

if __name__ == "__main__":
    unittest.main()
