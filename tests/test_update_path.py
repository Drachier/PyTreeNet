import unittest

import pytreenet as ptn

class TestUpdatePathSimple(unittest.TestCase):
    def setUp(self):
        tree = ptn.random_small_ttns()
        self.pathfinder = ptn.TDVPUpdatePathFinder(tree)

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
        tree = ptn.random_big_ttns_two_root_children()
        self.pathfinder = ptn.TDVPUpdatePathFinder(tree)

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

if __name__ == "__main__":
    unittest.main()
