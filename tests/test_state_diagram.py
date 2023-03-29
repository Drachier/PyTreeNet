import unittest

import pytreenet as ptn

class TestStateDiagram(unittest.TestCase):

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

    def test_from_single_term(self):
        term = {"site1": "1", "site2": "2", "site3": "3", "site4": "4", "site5": "5", "site6": "6", "site7": "7"}

        sd = ptn.StateDiagram.from_single_term(term, self.ref_tree)

        self.assertEqual(6, len(sd.vertex_colls))
        self.assertEqual(7, len(sd.hyperedge_colls))

        for site in sd.vertex_colls:
            self.assertEqual(1, len(sd.vertex_colls[site].contained_vertices))
        for site in sd.hyperedge_colls:
            self.assertEqual(1, len(sd.hyperedge_colls[site].contained_hyperedges))

    def test_find_path_with_origin_single_term(self):
        term = {"site1": "1", "site2": "2", "site3": "3", "site4": "4", "site5": "5", "site6": "6", "site7": "7"}

        sd = ptn.StateDiagram.from_single_term(term, self.ref_tree)

        origin = sd.vertex_colls[("site1","site2")].contained_vertices[0]
        path1_org, path2_org = sd.find_path_with_origin(origin)

        self.assertEqual(1, len(path1_org.targets))
        count = 0
        current_element = path1_org
        while len(current_element.targets) > 0:
            current_element = current_element.targets[0]
            count += 1
        self.assertEqual(5, count)

        self.assertEqual(1, len(path2_org.targets))
        count = 0
        current_element = path2_org
        while len(current_element.targets) > 0:
            current_element = current_element.targets[0]
            count += 1
        self.assertEqual(3, count)

if __name__ == "__main__":
    unittest.main()
