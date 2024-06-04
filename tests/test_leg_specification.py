import unittest

import pytreenet as ptn
from pytreenet.random import random_small_ttns

class TestLegSpecificationInit(unittest.TestCase):

    def test_empty_init(self):
        leg_spec = ptn.LegSpecification(None, None, None)
        self.assertIsNone(leg_spec.parent_leg)
        self.assertEqual(leg_spec.child_legs, [])
        self.assertEqual(leg_spec.open_legs, [])
        self.assertIsNone(leg_spec.node)
        self.assertFalse(leg_spec.is_root)

    def test_init(self):
        parent_leg = "parent"
        child_legs = ["child1", "child2"]
        open_legs = [1, 2]
        node = ptn.Node(identifier="node")
        leg_spec = ptn.LegSpecification(parent_leg, child_legs, open_legs,
                                        node)
        self.assertEqual(leg_spec.parent_leg, parent_leg)
        self.assertEqual(leg_spec.child_legs, child_legs)
        self.assertEqual(leg_spec.open_legs, open_legs)
        self.assertEqual(leg_spec.node, node)
        self.assertFalse(leg_spec.is_root)

class TestLegSpecificationEquality(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = [ptn.Node(identifier="node1"),
                      ptn.Node(identifier="node2")]
        self.parent_id = "parent"
        self.child_ids = ["child1","child2","child3"]

    def test_all_equal(self):
        leg_spec1 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0,1])
        leg_spec2 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0,1])
        self.assertEqual(leg_spec1,leg_spec2)

    def test_parent_not_equal(self):
        leg_spec1 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0,1])
        leg_spec2 = ptn.LegSpecification("other", self.child_ids,
                                         open_legs = [0,1])
        self.assertNotEqual(leg_spec1,leg_spec2)

    def test_children_not_equal(self):
        leg_spec1 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0,1])
        leg_spec2 = ptn.LegSpecification(self.parent_id,
                                         ["a","b","c"],
                                         open_legs = [0,1])
        self.assertNotEqual(leg_spec1,leg_spec2)

    def test_children_sublist(self):
        leg_spec1 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0,1])
        leg_spec2 = ptn.LegSpecification(self.parent_id,
                                         ["child1","child2"],
                                         open_legs = [0,1])
        self.assertNotEqual(leg_spec1,leg_spec2)

    def test_open_legs_not_equal(self):
        leg_spec1 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0,1])
        leg_spec2 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [2,3])
        self.assertNotEqual(leg_spec1,leg_spec2)

    def test_open_legs_sublist(self):
        leg_spec1 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0,1])
        leg_spec2 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0])
        self.assertNotEqual(leg_spec1,leg_spec2)

    def test_equal_nodes(self):
        leg_spec1 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0,1],
                                         node=self.nodes[0])
        leg_spec2 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0,1],
                                         node=self.nodes[0])
        self.assertEqual(leg_spec1,leg_spec2)

    def test_different_nodes(self):
        leg_spec1 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0,1],
                                         node=self.nodes[0])
        leg_spec2 = ptn.LegSpecification(self.parent_id, self.child_ids,
                                         open_legs = [0,1],
                                         node=self.nodes[1])
        self.assertNotEqual(leg_spec1,leg_spec2)

    def test_empty_equality(self):
        leg_spec1 = ptn.LegSpecification(None, None, None)
        leg_spec2 = ptn.LegSpecification(None, None, None)
        self.assertEqual(leg_spec1, leg_spec2)

class TestLegSpecMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = random_small_ttns().nodes

    def test_find_leg_values_root(self):
        leg_spec = ptn.LegSpecification(None,["c1","c2"],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,[],[],
                                        node=self.nodes["root"])
        self.assertEqual([0,1,2],leg_spec.find_leg_values())
        self.assertEqual([],leg_spec2.find_leg_values())

    def test_find_leg_values_root_split_c1(self):
        leg_spec = ptn.LegSpecification(None,["c2"],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,["c1"],[],
                                        node=self.nodes["root"])
        self.assertEqual([1,2],leg_spec.find_leg_values())
        self.assertEqual([0],leg_spec2.find_leg_values())

    def test_find_leg_values_root_split_c2(self):
        leg_spec = ptn.LegSpecification(None,["c1"],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,["c2"],[],
                                        node=self.nodes["root"])
        self.assertEqual([0,2],leg_spec.find_leg_values())
        self.assertEqual([1],leg_spec2.find_leg_values())

    def test_find_leg_values_root_split_open(self):
        leg_spec = ptn.LegSpecification(None,[],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,["c1","c2"],[],
                                        node=self.nodes["root"])
        self.assertEqual([2],leg_spec.find_leg_values())
        self.assertEqual([0,1],leg_spec2.find_leg_values())

    def test_find_leg_values_root_wrong_order(self):
        leg_spec = ptn.LegSpecification(None,["c2","c1"],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,[],[],
                                        node=self.nodes["root"])
        self.assertEqual([1,0,2],leg_spec.find_leg_values())
        self.assertEqual([],leg_spec2.find_leg_values())

    def test_find_leg_values_root_split_open_wrong_order(self):
        leg_spec = ptn.LegSpecification(None,[],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,["c2","c1"],[],
                                        node=self.nodes["root"])
        self.assertEqual([2],leg_spec.find_leg_values())
        self.assertEqual([1,0],leg_spec2.find_leg_values())

    def test_find_leg_values_leaf(self):
        leg_spec = ptn.LegSpecification("root",[],[1],
                                        node=self.nodes["c1"])
        leg_spec2 = ptn.LegSpecification(None,[],[],
                                        node=self.nodes["c1"])
        self.assertEqual([0,1],leg_spec.find_leg_values())
        self.assertEqual([],leg_spec2.find_leg_values())

    def test_find_leg_values_leaf_split(self):
        leg_spec = ptn.LegSpecification("root",[],[],
                                        node=self.nodes["c1"])
        leg_spec2 = ptn.LegSpecification(None,[],[1],
                                        node=self.nodes["c1"])
        self.assertEqual([0],leg_spec.find_leg_values())
        self.assertEqual([1],leg_spec2.find_leg_values())

    def test_find_all_neighbour_ids_root(self):
        leg_spec = ptn.LegSpecification(None,["c1","c2"],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,[],[],
                                        node=self.nodes["root"])
        self.assertEqual(["c1","c2"],leg_spec.find_all_neighbour_ids())
        self.assertEqual([],leg_spec2.find_all_neighbour_ids())

    def test_find_all_neighbour_ids_root_split_c1(self):
        leg_spec = ptn.LegSpecification(None,["c2"],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,["c1"],[],
                                        node=self.nodes["root"])
        self.assertEqual(["c2"],leg_spec.find_all_neighbour_ids())
        self.assertEqual(["c1"],leg_spec2.find_all_neighbour_ids())

    def test_find_all_neighbour_ids_root_split_c2(self):
        leg_spec = ptn.LegSpecification(None,["c1"],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,["c2"],[],
                                        node=self.nodes["root"])
        self.assertEqual(["c1"],leg_spec.find_all_neighbour_ids())
        self.assertEqual(["c2"],leg_spec2.find_all_neighbour_ids())

    def test_find_all_neighbour_ids_root_split_open(self):
        leg_spec = ptn.LegSpecification(None,[],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,["c1","c2"],[],
                                        node=self.nodes["root"])
        self.assertEqual([],leg_spec.find_all_neighbour_ids())
        self.assertEqual(["c1","c2"],leg_spec2.find_all_neighbour_ids())

    def test_find_all_neighbour_ids_root_wrong_order(self):
        leg_spec = ptn.LegSpecification(None,["c2","c1"],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,[],[],
                                        node=self.nodes["root"])
        self.assertEqual(["c2","c1"],leg_spec.find_all_neighbour_ids())
        self.assertEqual([],leg_spec2.find_all_neighbour_ids())

    def test_find_all_neighbour_ids_root_split_open_wrong_order(self):
        leg_spec = ptn.LegSpecification(None,[],[2],
                                        node=self.nodes["root"])
        leg_spec2 = ptn.LegSpecification(None,["c2","c1"],[],
                                        node=self.nodes["root"])
        self.assertEqual([],leg_spec.find_all_neighbour_ids())
        self.assertEqual(["c2","c1"],leg_spec2.find_all_neighbour_ids())

    def test_find_all_neighbour_ids_leaf(self):
        leg_spec = ptn.LegSpecification("root",[],[1],
                                        node=self.nodes["c1"])
        leg_spec2 = ptn.LegSpecification(None,[],[],
                                        node=self.nodes["c1"])
        self.assertEqual(["root"],leg_spec.find_all_neighbour_ids())
        self.assertEqual([],leg_spec2.find_all_neighbour_ids())

    def test_find_all_neighbour_ids_leaf_split(self):
        leg_spec = ptn.LegSpecification("root",[],[],
                                        node=self.nodes["c1"])
        leg_spec2 = ptn.LegSpecification(None,[],[1],
                                        node=self.nodes["c1"])
        self.assertEqual(["root"],leg_spec.find_all_neighbour_ids())
        self.assertEqual([],leg_spec2.find_all_neighbour_ids())

if __name__ == "__main__":
    unittest.main()
