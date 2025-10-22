"""
This module contains tests for the random generation of tensor networks (TTNs).
"""
import unittest

from pytreenet.random.random_special_ttns import (random_mps,
                                                  random_binary_state,
                                                  random_ftps,
                                                  random_tstar_state)
from pytreenet.random.random_ttns import (random_big_ttns,
                                          random_small_ttns,
                                          random_big_ttns_two_root_children,
                                          random_like,
                                          RandomTTNSMode)

class TestRandomLike(unittest.TestCase):
    """
    Tests the `random_like` function for generating random TTNs from a TTN.
    """

    def test_random_like_small_ttns(self):
        """
        Test generating random TTNs similar to small TTNs.
        """
        ttns = random_small_ttns()
        found = random_like(ttns)
        self.assertTrue(found.same_hierarchy_as(ttns))
        self.assertTrue(found.same_dimensions_as(ttns))

    def test_random_like_small_ttns_smaller_bd(self):
        """
        Test generating random TTNs similar to small TTNs with smaller bond
        dimension.
        """
        ttns = random_small_ttns()
        found = random_like(ttns, bond_dim=4)
        self.assertTrue(found.same_hierarchy_as(ttns))
        self.assertFalse(found.same_dimensions_as(ttns))
        self.assertTrue(found.max_bond_dim() == 4)
        self.assertTrue(found.same_open_dimensions_as(ttns))

    def test_random_like_big_ttns(self):
        """
        Test generating random TTNs similar to big TTNs.
        """
        ttns = random_big_ttns()
        found = random_like(ttns)
        self.assertTrue(found.same_hierarchy_as(ttns))
        self.assertTrue(found.same_dimensions_as(ttns))

    def test_random_like_big_ttns_smaller_bd(self):
        """
        Test generating random TTNs similar to big TTNs with smaller bond
        dimension.
        """
        ttns = random_big_ttns(mode=RandomTTNSMode.DIFFVIRT)
        found = random_like(ttns, bond_dim=3)
        self.assertTrue(found.same_hierarchy_as(ttns))
        self.assertTrue(found.max_bond_dim() == 3)
        self.assertTrue(found.same_open_dimensions_as(ttns))

    def test_random_like_big_ttns_two_root_children(self):
        """
        Test generating random TTNs similar to big TTNs with two root children.
        """
        ttns = random_big_ttns_two_root_children()
        found = random_like(ttns)
        self.assertTrue(found.same_hierarchy_as(ttns))
        self.assertTrue(found.same_dimensions_as(ttns))

    def test_random_mps(self):
        """
        Test generating random MPS.
        """
        ttns = random_mps(10, 2, 20)
        found = random_like(ttns)
        self.assertTrue(found.same_hierarchy_as(ttns))
        self.assertTrue(found.same_dimensions_as(ttns))

    def test_random_binary_state(self):
        """
        Test generating random binary tree states.
        """
        ttns = random_binary_state(2, 2, 15)
        found = random_like(ttns)
        self.assertTrue(found.same_hierarchy_as(ttns))
        self.assertTrue(found.same_dimensions_as(ttns))

    def test_random_tstar_state(self):
        """
        Test generating random T-star states.
        """
        ttns = random_tstar_state(5, 3, 10)
        found = random_like(ttns)
        self.assertTrue(found.same_hierarchy_as(ttns))
        self.assertTrue(found.same_dimensions_as(ttns))

    def test_random_ftps(self):
        """
        Test generating random FTP states.
        """
        ttns = random_ftps(4, 2, 8)
        found = random_like(ttns)
        self.assertTrue(found.same_hierarchy_as(ttns))
        self.assertTrue(found.same_dimensions_as(ttns))

if __name__ == '__main__':
    unittest.main()