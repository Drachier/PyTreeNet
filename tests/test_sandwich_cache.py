import unittest

from numpy import allclose

from pytreenet.contractions.state_operator_contraction import (contract_leaf,
                                                               contract_subtrees_using_dictionary)

from pytreenet.contractions.sandwich_caching import (SandwichCache,
                                                     _find_caching_path)

from pytreenet.random.random_ttns import (random_small_ttns,
                                            random_big_ttns_two_root_children)
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)

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

    def test_find_caching_path_7(self):
        ref_tree = random_big_ttns_two_root_children()
        caching_path, next_node_id_dict = _find_caching_path(ref_tree, "site7")
        correct_path = ["site2","site4","site5","site3","site1","site0","site6","site7"]
        self.assertEqual(correct_path, caching_path)
        correct_dict = {"site6": "site7", "site0": "site6",
                        "site2": "site1", "site1": "site0",
                        "site4": "site3", "site3": "site1",
                        "site5": "site3"}
        self.assertEqual(correct_dict, next_node_id_dict)

class TestUpdateTreeCache(unittest.TestCase):

    def test_update_tree_cache_simple(self):
        """
        Tests caching form c2 to c1 in a simple tree.
        """
        ref_ttns, ref_ttno = small_ttns_and_ttno()
        cache = SandwichCache(ref_ttns, ref_ttno)
        node_id = "c2"
        state_node, state_tensor = ref_ttns[node_id]
        op_node, op_tensor = ref_ttno[node_id]
        ref_tensor = contract_leaf(state_node, state_tensor,
                                   op_node, op_tensor)
        cache.update_tree_cache("c2", "root")
        self.assertTrue(allclose(ref_tensor,
                                 cache.get_entry("c2", "root")))
        node_id = "root"
        state_node, state_tensor = ref_ttns[node_id]
        op_node, op_tensor = ref_ttno[node_id]
        ref_tensor = contract_subtrees_using_dictionary("c1",
                                                        state_node, state_tensor,
                                                        op_node, op_tensor,
                                                        cache)

        cache.update_tree_cache("root", "c1")
        self.assertTrue(allclose(ref_tensor,
                                 cache.get_entry("root", "c1")))

    def test_update_tree_cache_complicated(self):
        """
        Tests caching from 7 to 5 in a complicated tree.
        """
        ref_ttns, ref_ttno = big_ttns_and_ttno()
        cache = SandwichCache(ref_ttns, ref_ttno)
        node_id = "site7"
        state_node, state_tensor = ref_ttns[node_id]
        op_node, op_tensor = ref_ttno[node_id]
        ref_tensor = contract_leaf(state_node, state_tensor,
                                   op_node, op_tensor)
        cache.update_tree_cache("site7", "site6")
        self.assertTrue(allclose(ref_tensor,
                                 cache.get_entry("site7", "site6")))
        node_id = "site6"
        state_node, state_tensor = ref_ttns[node_id]
        op_node, op_tensor = ref_ttno[node_id]
        ref_tensor = contract_subtrees_using_dictionary("site0",
                                                        state_node, state_tensor,
                                                        op_node, op_tensor,
                                                        cache)
        cache.update_tree_cache("site6", "site0")
        self.assertTrue(allclose(ref_tensor,
                                 cache.get_entry("site6", "site0")))
        node_id = "site0"
        state_node, state_tensor = ref_ttns[node_id]
        op_node, op_tensor = ref_ttno[node_id]
        ref_tensor = contract_subtrees_using_dictionary("site1",
                                                        state_node, state_tensor,
                                                        op_node, op_tensor,
                                                        cache)
        cache.update_tree_cache("site0", "site1")
        self.assertTrue(allclose(ref_tensor,
                                 cache.get_entry("site0", "site1")))
        node_id = "site2"
        state_node, state_tensor = ref_ttns[node_id]
        op_node, op_tensor = ref_ttno[node_id]
        ref_tensor = contract_leaf(state_node, state_tensor,
                                   op_node, op_tensor)
        cache.update_tree_cache("site2", "site1")
        self.assertTrue(allclose(ref_tensor,
                                 cache.get_entry("site2", "site1")))
        node_id = "site1"
        state_node, state_tensor = ref_ttns[node_id]
        op_node, op_tensor = ref_ttno[node_id]
        ref_tensor = contract_subtrees_using_dictionary("site3",
                                                        state_node, state_tensor,
                                                        op_node, op_tensor,
                                                        cache)
        cache.update_tree_cache("site1", "site3")
        self.assertTrue(allclose(ref_tensor,
                                 cache.get_entry("site1", "site3")))
        node_id = "site4"
        state_node, state_tensor = ref_ttns[node_id]
        op_node, op_tensor = ref_ttno[node_id]
        ref_tensor = contract_leaf(state_node, state_tensor,
                                   op_node, op_tensor)
        cache.update_tree_cache("site4", "site3")
        self.assertTrue(allclose(ref_tensor,
                                 cache.get_entry("site4", "site3")))
        node_id = "site3"
        state_node, state_tensor = ref_ttns[node_id]
        op_node, op_tensor = ref_ttno[node_id]
        ref_tensor = contract_subtrees_using_dictionary("site5",
                                                        state_node, state_tensor,
                                                        op_node, op_tensor,
                                                        cache)
        cache.update_tree_cache("site3", "site5")
        self.assertTrue(allclose(ref_tensor,
                                 cache.get_entry("site3", "site5")))

class TestSandwichCacheInitButOne(unittest.TestCase):

    def test_init_cache_but_one_simple_root(self):
        ref_ttns, ref_ttno = small_ttns_and_ttno()
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("c2", "root")
        ref_cache.update_tree_cache("c1", "root")
        found_cache = SandwichCache.init_cache_but_one(ref_ttns,
                                                        ref_ttno,
                                                        "root")
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_init_cache_but_one_simple_c1(self):
        ref_ttns, ref_ttno = small_ttns_and_ttno()
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("c2", "root")
        ref_cache.update_tree_cache("root", "c1")
        found_cache = SandwichCache.init_cache_but_one(ref_ttns,
                                                        ref_ttno,
                                                        "c1")
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_init_cache_but_one_simple_c2(self):
        ref_ttns, ref_ttno = small_ttns_and_ttno()
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("c1", "root")
        ref_cache.update_tree_cache("root", "c2")
        found_cache = SandwichCache.init_cache_but_one(ref_ttns,
                                                        ref_ttno,
                                                        "c2")
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_init_cache_but_one_complicated_0(self):
        ref_ttns, ref_ttno = big_ttns_and_ttno()
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("site7", "site6")
        ref_cache.update_tree_cache("site6", "site0")
        ref_cache.update_tree_cache("site4", "site3")
        ref_cache.update_tree_cache("site5", "site3")
        ref_cache.update_tree_cache("site3", "site1")
        ref_cache.update_tree_cache("site2", "site1")
        ref_cache.update_tree_cache("site1", "site0")
        found_cache = SandwichCache.init_cache_but_one(ref_ttns,
                                                        ref_ttno,
                                                        "site0")
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_init_cache_but_one_complicated_1(self):
        ref_ttns, ref_ttno = big_ttns_and_ttno()
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("site7", "site6")
        ref_cache.update_tree_cache("site6", "site0")
        ref_cache.update_tree_cache("site0", "site1")
        ref_cache.update_tree_cache("site2", "site1")
        ref_cache.update_tree_cache("site4", "site3")
        ref_cache.update_tree_cache("site5", "site3")
        ref_cache.update_tree_cache("site3", "site1")
        found_cache = SandwichCache.init_cache_but_one(ref_ttns,
                                                        ref_ttno,
                                                        "site1")
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_init_cache_but_one_complicated_2(self):
        ref_ttns, ref_ttno = big_ttns_and_ttno()
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("site7", "site6")
        ref_cache.update_tree_cache("site6", "site0")
        ref_cache.update_tree_cache("site0", "site1")
        ref_cache.update_tree_cache("site4", "site3")
        ref_cache.update_tree_cache("site5", "site3")
        ref_cache.update_tree_cache("site3", "site1")
        ref_cache.update_tree_cache("site1", "site2")
        found_cache = SandwichCache.init_cache_but_one(ref_ttns,
                                                        ref_ttno,
                                                        "site2")
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_init_cache_but_one_complicated_3(self):
        ref_ttns, ref_ttno = big_ttns_and_ttno()
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("site7", "site6")
        ref_cache.update_tree_cache("site6", "site0")
        ref_cache.update_tree_cache("site0", "site1")
        ref_cache.update_tree_cache("site2", "site1")
        ref_cache.update_tree_cache("site1", "site3")
        ref_cache.update_tree_cache("site4", "site3")
        ref_cache.update_tree_cache("site5", "site3")
        found_cache = SandwichCache.init_cache_but_one(ref_ttns,
                                                        ref_ttno,
                                                        "site3")
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_init_cache_but_one_complicated_4(self):
        ref_ttns, ref_ttno = big_ttns_and_ttno()
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("site7", "site6")
        ref_cache.update_tree_cache("site6", "site0")
        ref_cache.update_tree_cache("site0", "site1")
        ref_cache.update_tree_cache("site2", "site1")
        ref_cache.update_tree_cache("site1", "site3")
        ref_cache.update_tree_cache("site5", "site3")
        ref_cache.update_tree_cache("site3", "site4")
        found_cache = SandwichCache.init_cache_but_one(ref_ttns,
                                                        ref_ttno,
                                                        "site4")
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_init_cache_but_one_complicated_5(self):
        ref_ttns, ref_ttno = big_ttns_and_ttno()
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("site7", "site6")
        ref_cache.update_tree_cache("site6", "site0")
        ref_cache.update_tree_cache("site0", "site1")
        ref_cache.update_tree_cache("site2", "site1")
        ref_cache.update_tree_cache("site1", "site3")
        ref_cache.update_tree_cache("site4", "site3")
        ref_cache.update_tree_cache("site3", "site5")
        found_cache = SandwichCache.init_cache_but_one(ref_ttns,
                                                        ref_ttno,
                                                        "site5")
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_init_cache_but_one_complicated_6(self):
        ref_ttns, ref_ttno = big_ttns_and_ttno()
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("site7", "site6")
        ref_cache.update_tree_cache("site4", "site3")
        ref_cache.update_tree_cache("site5", "site3")
        ref_cache.update_tree_cache("site3", "site1")
        ref_cache.update_tree_cache("site2", "site1")
        ref_cache.update_tree_cache("site1", "site0")
        ref_cache.update_tree_cache("site0", "site6")
        found_cache = SandwichCache.init_cache_but_one(ref_ttns,
                                                        ref_ttno,
                                                        "site6")
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_init_cache_but_one_complicated_7(self):
        ref_ttns, ref_ttno = big_ttns_and_ttno()
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("site4", "site3")
        ref_cache.update_tree_cache("site5", "site3")
        ref_cache.update_tree_cache("site3", "site1")
        ref_cache.update_tree_cache("site2", "site1")
        ref_cache.update_tree_cache("site1", "site0")
        ref_cache.update_tree_cache("site0", "site6")
        ref_cache.update_tree_cache("site6", "site7")
        found_cache = SandwichCache.init_cache_but_one(ref_ttns,
                                                        ref_ttno,
                                                        "site7")
        self.assertTrue(ref_cache.close_to(found_cache))
