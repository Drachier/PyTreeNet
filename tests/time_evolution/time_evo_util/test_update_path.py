import unittest

from pytreenet.random import (random_big_ttns,
                              random_small_ttns,
                              random_big_ttns_two_root_children)
from pytreenet.time_evolution.time_evo_util import (
                              TDVPUpdatePathFinder_LeafToRoot,
                              TDVPUpdatePathFinder_LeafToLeaf)

class TestUpdatePathSimple(unittest.TestCase):
    def setUp(self):
        tree = random_small_ttns()
        self.pathfinder = TDVPUpdatePathFinder_LeafToRoot(tree)

    def test_find_start_node_id(self):
        self.assertEqual("c1", self.pathfinder.find_start_node_id())
        self.assertEqual("c1", self.pathfinder.start)

    def test_main_path_init(self):
        self.assertEqual(["c1", "root"], self.pathfinder.main_path)

    def test_path_for_branch(self):
        found_path = self.pathfinder.path_for_branch("c1")
        self.assertEqual(["c1"], found_path)

    def test_find_furthest_non_visited_leaf(self):
        found_leaf = self.pathfinder.find_furthest_non_visited_leaf(["c1","root"])
        self.assertEqual("c2", found_leaf)

    def test_find_main_path_down_from_root(self):
        found_path = self.pathfinder.find_main_path_down_from_root(["c1"])
        self.assertEqual(["root", "c2"], found_path)

    def test_branch_downwards_origin_is_root(self):
        found_path = self.pathfinder._branch_downwards_origin_is_root(["root", "c2"])
        self.assertEqual(["root"], found_path)

    def test_branch_path_downwards(self):
        found_path = self.pathfinder._branch_path_downwards("c2", ["root","c2"])
        self.assertEqual(["c2"], found_path)

    def test_path_down_from_root(self):
        found_path = self.pathfinder.path_down_from_root(["c1"])
        self.assertEqual(["root", "c2"], found_path)

    def test_find_path(self):
        self.assertEqual(["c1", "root", "c2"], self.pathfinder.find_path())

class TestUpdatePathComplicated(unittest.TestCase):
    def setUp(self):
        tree = random_big_ttns_two_root_children()
        self.pathfinder = TDVPUpdatePathFinder_LeafToRoot(tree)

    def test_find_start_node_id(self):
        self.assertEqual("site4", self.pathfinder.find_start_node_id())
        self.assertEqual("site4", self.pathfinder.start)

    def test_main_path_init(self):
        correct_path = ["site4","site3","site1","site0"]
        self.assertEqual(correct_path, self.pathfinder.main_path)

    def test_path_for_branch_site4(self):
        found_path = self.pathfinder.path_for_branch("site4")
        self.assertEqual(["site4"], found_path)

    def test_path_for_branch_site3(self):
        found_path = self.pathfinder.path_for_branch("site3")
        correct_path = ["site5","site3"]
        self.assertEqual(correct_path, found_path)

    def test_path_for_branch_site1(self):
        found_path = self.pathfinder.path_for_branch("site1")
        correct_path = ["site2","site1"]
        self.assertEqual(correct_path, found_path)

    def test_path_for_branch_site0(self):
        found_path = self.pathfinder.path_for_branch("site0")
        correct_path = ["site7","site6","site0"]
        self.assertEqual(correct_path, found_path)

    def test_find_furthest_non_visited_leaf(self):
        visited_path = ["site4","site5","site3","site2","site1"]
        found_leaf = self.pathfinder.find_furthest_non_visited_leaf(visited_path)
        self.assertEqual("site7", found_leaf)

    def test_find_main_path_down_from_root(self):
        visited_path = ["site4","site5","site3","site2","site1"]
        found_path = self.pathfinder.find_main_path_down_from_root(visited_path)
        self.assertEqual(["site0","site6","site7"], found_path)

    def test_branch_downwards_origin_is_root(self):
        main_down_path = ["site0","site6","site7"]
        found_path = self.pathfinder._branch_downwards_origin_is_root(main_down_path)
        self.assertEqual(["site0"], found_path)

    def test_branch_path_downwards6(self):
        main_down_path = ["site0","site6","site7"]
        found_path = self.pathfinder._branch_path_downwards("site6", main_down_path)
        self.assertEqual(["site6"], found_path)

    def test_branch_path_downwards7(self):
        main_down_path = ["site0","site6","site7"]
        found_path = self.pathfinder._branch_path_downwards("site7", main_down_path)
        self.assertEqual(["site7"], found_path)

    def test_path_down_from_root(self):
        visited_path = ["site4","site5","site3","site2","site1"]
        found_path = self.pathfinder.path_down_from_root(visited_path)
        correct_path = ["site0","site6","site7"]
        self.assertEqual(correct_path, found_path)

    def test_find_path(self):
        correct_path = ["site4","site5","site3","site2",
                        "site1","site0","site6","site7"]
        self.assertEqual(correct_path, self.pathfinder.find_path())

class TestLeafToLeafBig_1(unittest.TestCase):
    """
    Tests TDVPUpdatePathFinder_LeafToLeaf on random_big_ttns().
        - All leaves: ['site3', 'site5', 'site7', 'site8']
        - _find_two_diameter_leaves() -> ('site3', 'site5')
        - start = site3, end = site5
        - main_path = ['site3','site2','site1','site4','site5']
        - find_path() -> ['site3', 'site2', 'site7', 'site8', 'site6', 'site1', 'site4', 'site5']
    """
    @classmethod
    def setUpClass(cls):
        cls.tree = random_big_ttns()
        cls.pathfinder = TDVPUpdatePathFinder_LeafToLeaf(cls.tree)

    def test_find_two_diameter_leaves(self):
        expected = ('site3', 'site5')
        found = self.pathfinder._find_two_diameter_leaves()
        self.assertEqual(expected, found)

    def test_start(self):
        self.assertEqual("site3", self.pathfinder.start)

    def test_end(self):
        self.assertEqual("site5", self.pathfinder.end)

    def test_main_path(self):
        expected = ['site3','site2','site1','site4','site5']
        self.assertEqual(expected, self.pathfinder.main_path)

    def test_visit_offpath_subtree_parents_site3(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_parents("site3", visited)
        self.assertEqual([], result)

    def test_visit_offpath_subtree_children_site3(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_children("site3", visited)
        self.assertEqual([], result)

    def test_subtree_path_rec_site3(self):
        visited = set()
        result = self.pathfinder._subtree_path_rec("site3", visited)
        self.assertEqual([], result)

    def test_visit_offpath_subtree_parents_site2(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_parents("site2", visited)
        self.assertEqual([], result)

    def test_visit_offpath_subtree_children_site2(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_children("site2", visited)
        self.assertEqual([], result)

    def test_subtree_path_rec_site2(self):
        visited = set()
        result = self.pathfinder._subtree_path_rec("site2", visited)
        self.assertEqual([], result)

    def test_visit_offpath_subtree_parents_site1(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_parents("site1", visited)
        self.assertEqual([], result)

    def test_visit_offpath_subtree_children_site1(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_children("site1", visited)
        self.assertEqual(["site7","site8","site6"], result)

    def test_subtree_path_rec_site1(self):
        visited = set()
        result = self.pathfinder._subtree_path_rec("site1", visited)
        self.assertEqual([], result)

    def test_visit_offpath_subtree_parents_site4(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_parents("site4", visited)
        self.assertEqual([], result)

    def test_visit_offpath_subtree_children_site4(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_children("site4", visited)
        self.assertEqual([], result)

    def test_subtree_path_rec_site4(self):
        visited = set()
        result = self.pathfinder._subtree_path_rec("site4", visited)
        self.assertEqual([], result)

    def test_visit_offpath_subtree_parents_site5(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_parents("site5", visited)
        self.assertEqual([], result)

    def test_visit_offpath_subtree_children_site5(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_children("site5", visited)
        self.assertEqual([], result)

    def test_subtree_path_rec_site5(self):
        visited = set()
        result = self.pathfinder._subtree_path_rec("site5", visited)
        self.assertEqual([], result)

    def test_find_path(self):
        # final path
        expected = ['site3','site2','site7','site8','site6','site1','site4','site5']
        found = self.pathfinder.find_path()
        self.assertEqual(expected, found)

class TestLeafToLeafBig_2(unittest.TestCase):
    """
    Tests TDVPUpdatePathFinder_LeafToLeaf on random_big_ttns_two_root_children().

        All leaves: ['site2', 'site4', 'site5', 'site7']
        _find_two_diameter_leaves() -> ('site4', 'site7')
        start = site4
        end   = site7
        main_path = ['site4','site3','site1','site0','site6','site7']
        find_path() -> ['site4', 'site5', 'site3', 'site2','site1','site0','site6','site7']
    """
    @classmethod
    def setUpClass(cls):
        cls.tree = random_big_ttns_two_root_children()
        cls.pathfinder = TDVPUpdatePathFinder_LeafToLeaf(cls.tree)

    def test_find_two_diameter_leaves(self):
        expected = ('site4','site7')
        found = self.pathfinder._find_two_diameter_leaves()
        self.assertEqual(expected, found)

    def test_start(self):
        self.assertEqual("site4", self.pathfinder.start)

    def test_end(self):
        self.assertEqual("site7", self.pathfinder.end)

    def test_main_path(self):
        expected = ["site4","site3","site1","site0","site6","site7"]
        self.assertEqual(expected, self.pathfinder.main_path)

    def test_offpath_subtree_parents_site4(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_parents("site4", visited)
        self.assertEqual([], result)

    def test_offpath_subtree_children_site4(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_children("site4", visited)
        self.assertEqual([], result)

    def test_subtree_path_rec_site4(self):
        visited = set()
        result = self.pathfinder._subtree_path_rec("site4", visited)
        self.assertEqual([], result)

    def test_offpath_subtree_parents_site3(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_parents("site3", visited)
        self.assertEqual([], result)

    def test_offpath_subtree_children_site3(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_children("site3", visited)
        # from your logs: ['site5']
        self.assertEqual(["site5"], result)

    def test_subtree_path_rec_site3(self):
        visited = set()
        result = self.pathfinder._subtree_path_rec("site3", visited)
        self.assertEqual([], result)

    def test_offpath_subtree_parents_site1(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_parents("site1", visited)
        self.assertEqual([], result)

    def test_offpath_subtree_children_site1(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_children("site1", visited)
        self.assertEqual(["site2"], result)

    def test_subtree_path_rec_site1(self):
        visited = set()
        result = self.pathfinder._subtree_path_rec("site1", visited)
        self.assertEqual([], result)

    def test_offpath_subtree_parents_site0(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_parents("site0", visited)
        self.assertEqual([], result)

    def test_offpath_subtree_children_site0(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_children("site0", visited)
        self.assertEqual([], result)

    def test_subtree_path_rec_site0(self):
        visited = set()
        result = self.pathfinder._subtree_path_rec("site0", visited)
        self.assertEqual([], result)

    def test_offpath_subtree_parents_site6(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_parents("site6", visited)
        self.assertEqual([], result)

    def test_offpath_subtree_children_site6(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_children("site6", visited)
        self.assertEqual([], result)

    def test_subtree_path_rec_site6(self):
        visited = set()
        result = self.pathfinder._subtree_path_rec("site6", visited)
        self.assertEqual([], result)

    def test_offpath_subtree_parents_site7(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_parents("site7", visited)
        self.assertEqual([], result)

    def test_offpath_subtree_children_site7(self):
        visited = set()
        result = self.pathfinder._visit_offpath_subtree_children("site7", visited)
        self.assertEqual([], result)

    def test_subtree_path_rec_site7(self):
        visited = set()
        result = self.pathfinder._subtree_path_rec("site7", visited)
        self.assertEqual([], result)

    def test_find_path(self):
        # final path
        expected = ["site4","site5","site3","site2","site1","site0","site6","site7"]
        found = self.pathfinder.find_path()
        self.assertEqual(expected, found)

if __name__ == "__main__":
    unittest.main()
