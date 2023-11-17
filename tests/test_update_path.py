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


if __name__ == "__main__":
    unittest.main()