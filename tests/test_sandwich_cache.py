import unittest

from pytreenet.contractions.sandwich_caching import (SandwichCache,
                                                     _find_caching_path)

from pytreenet.random.random_ttns import (random_small_ttns,
                                            random_big_ttns_two_root_children)

class TestChachingPathFinding(unittest.TestCase):

    def test_find_caching_path_c1(self):
        ref_tree = random_small_ttns()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "c1")
        self.assertEqual(["c2", "root", "c1"], caching_path)
        self.assertEqual({"c2": "root", "root": "c1"}, next_node_id_dict)

    def test_find_caching_path_c2(self):
        ref_tree = random_small_ttns()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "c2")
        self.assertEqual(["c1", "root", "c2"], caching_path)
        self.assertEqual({"c1": "root", "root": "c2"}, next_node_id_dict)

    def test_find_caching_path_root(self):
        ref_tree = random_small_ttns()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "root")
        self.assertEqual(["c1", "c2", "root"], caching_path)
        self.assertEqual({"c1": "root", "c2": "root"}, next_node_id_dict)

    def test_find_caching_path_0(self):
        ref_tree = random_big_ttns_two_root_children()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "site0")
        correct_path = ["site2","site4","site5","site3",
                        "site1","site7","site6","site0"]
        self.assertEqual(correct_path, caching_path)
        correct_dict = {"site7": "site6", "site6": "site0",
                        "site2": "site1", "site1": "site0",
                        "site5": "site3", "site3": "site1",
                        "site4": "site3"}
        self.assertEqual(correct_dict, next_node_id_dict)

    def test_find_caching_path_1(self):
        ref_tree = random_big_ttns_two_root_children()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "site1")
        correct_path = ["site7","site6","site0","site2",
                        "site4","site5","site3","site1"]
        self.assertEqual(correct_path, caching_path)
        correct_dict = {"site7": "site6", "site6": "site0",
                        "site2": "site1", "site0": "site1",
                        "site5": "site3", "site3": "site1",
                        "site4": "site3"}
        self.assertEqual(correct_dict, next_node_id_dict)

    def test_find_caching_path_2(self):
        ref_tree = random_big_ttns_two_root_children()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "site2")
        correct_path = ["site7","site6","site0","site4",
                        "site5","site3","site1","site2"]
        self.assertEqual(correct_path, caching_path)
        correct_dict = {"site7": "site6", "site6": "site0",
                        "site1": "site2", "site0": "site1",
                        "site5": "site3", "site3": "site1",
                        "site4": "site3"}
        self.assertEqual(correct_dict, next_node_id_dict)

    def test_find_caching_path_3(self):
        ref_tree = random_big_ttns_two_root_children()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "site3")
        correct_path = ["site7","site6","site0","site2",
                        "site1","site4","site5","site3"]
        self.assertEqual(correct_path, caching_path)
        correct_dict = {"site7": "site6", "site6": "site0",
                        "site2": "site1", "site0": "site1",
                        "site5": "site3", "site1": "site3",
                        "site4": "site3"}
        self.assertEqual(correct_dict, next_node_id_dict)

    def test_find_caching_path_4(self):
        ref_tree = random_big_ttns_two_root_children()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "site4")
        correct_path = ["site7","site6","site0","site2",
                        "site1","site5","site3","site4"]
        self.assertEqual(correct_path, caching_path)
        correct_dict = {"site7": "site6", "site6": "site0",
                        "site2": "site1", "site0": "site1",
                        "site5": "site3", "site1": "site3",
                        "site3": "site4"}
        self.assertEqual(correct_dict, next_node_id_dict)

    def test_find_caching_path_5(self):
        ref_tree = random_big_ttns_two_root_children()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "site5")
        correct_path = ["site7","site6","site0","site2",
                        "site1","site4","site3","site5"]
        self.assertEqual(correct_path, caching_path)
        correct_dict = {"site7": "site6", "site6": "site0",
                        "site2": "site1", "site0": "site1",
                        "site4": "site3", "site1": "site3",
                        "site3": "site5"}
        self.assertEqual(correct_dict, next_node_id_dict)

    def test_find_caching_path_6(self):
        ref_tree = random_big_ttns_two_root_children()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "site6")
        correct_path = ["site2","site4","site5","site3",
                        "site1","site0","site7","site6"]
        self.assertEqual(correct_path, caching_path)
        correct_dict = {"site7": "site6", "site0": "site6",
                        "site2": "site1", "site1": "site0",
                        "site4": "site3", "site3": "site1",
                        "site5": "site3"}
        self.assertEqual(correct_dict, next_node_id_dict)

    def test_find_caching_path_6(self):
        ref_tree = random_big_ttns_two_root_children()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "site7")
        correct_path = ["site2","site4","site5","site3","site1","site0","site6","site7"]
        self.assertEqual(correct_path, caching_path)
        correct_dict = {"site6": "site7", "site0": "site6",
                        "site2": "site1", "site1": "site0",
                        "site4": "site3", "site3": "site1",
                        "site5": "site3"}
        self.assertEqual(correct_dict, next_node_id_dict)