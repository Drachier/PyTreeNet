import unittest
from copy import deepcopy

from numpy import allclose

from pytreenet.core.leg_specification import LegSpecification

from pytreenet.random.random_ttns import (random_small_ttns)
from pytreenet.random.random_node import crandn

class TestReplaceSplittingSimple(unittest.TestCase):

    def test_split_c1(self):
        ttns = random_small_ttns()
        tensor_a = crandn((5,2))
        tensor_b = crandn((2,3))
        ref_a = deepcopy(tensor_a)
        ref_b = deepcopy(tensor_b)
        ida = "a"
        idb = "b"
        legs_a = LegSpecification("root", [], [])
        legs_b = LegSpecification(None, [], [1])
        ttns.split_node_replace("c1", tensor_a, tensor_b,
                                    ida, idb,
                                    legs_a, legs_b)
        self.assertTrue(ida in ttns.nodes)
        self.assertTrue(idb in ttns.nodes)
        self.assertTrue(ida in ttns.tensors)
        self.assertTrue(idb in ttns.tensors)
        self.assertTrue(allclose(ref_a, ttns.tensors[ida]))
        self.assertTrue(allclose(ref_b, ttns.tensors[idb]))
        self.assertTrue(ttns.is_child_of(ida, "root"))
        self.assertTrue(ttns.is_child_of(idb, ida))
        self.assertTrue(ttns.is_parent_of("root", ida))
        self.assertTrue(ttns.is_parent_of(ida, idb))

    def test_split_c1_flipped(self):
        ttns = random_small_ttns()
        tensor_a = crandn((2,5))
        tensor_b = crandn((3,2))
        ref_a = deepcopy(tensor_a)
        ref_b = deepcopy(tensor_b)
        ida = "a"
        idb = "b"
        legs_a = LegSpecification("root", [], [])
        legs_b = LegSpecification(None, [], [1])
        ttns.split_node_replace("c1", tensor_b, tensor_a,
                                    idb, ida,
                                    legs_b, legs_a)
        self.assertTrue(ida in ttns.nodes)
        self.assertTrue(idb in ttns.nodes)
        self.assertTrue(ida in ttns.tensors)
        self.assertTrue(idb in ttns.tensors)
        self.assertTrue(allclose(ref_a.transpose(1,0), ttns.tensors[ida]))
        self.assertTrue(allclose(ref_b.transpose(1,0), ttns.tensors[idb]))
        self.assertTrue(ttns.is_child_of(ida, "root"))
        self.assertTrue(ttns.is_child_of(idb, ida))
        self.assertTrue(ttns.is_parent_of("root", ida))
        self.assertTrue(ttns.is_parent_of(ida, idb))

    def test_split_root_c1andphys_together(self):
        """
        In this test one node has the physical leg and the leg to c1 together.
        """
        ttns = random_small_ttns()
        tensor_a = crandn((5,2,3))
        tensor_b = crandn((3,6))
        ref_a = deepcopy(tensor_a)
        ref_b = deepcopy(tensor_b)
        ida = "a"
        idb = "b"
        legs_a = LegSpecification(None, ["c1"], [2], is_root=True)
        legs_b = LegSpecification(None, ["c2"], [])
        ttns.split_node_replace("root", tensor_a, tensor_b,
                                    ida, idb,
                                    legs_a, legs_b)
        # A child order = b, c1
        self.assertTrue(ida in ttns.nodes)
        self.assertTrue(idb in ttns.nodes)
        self.assertTrue(ida in ttns.tensors)
        self.assertTrue(idb in ttns.tensors)
        self.assertTrue(allclose(ref_a.transpose(2,0,1), ttns.tensors[ida]))
        self.assertTrue(allclose(ref_b, ttns.tensors[idb]))
        self.assertTrue(ttns.is_child_of("c1", ida))
        self.assertTrue(ttns.is_parent_of(ida, "c1"))
        self.assertTrue(ttns.is_child_of("c2", idb))
        self.assertTrue(ttns.is_parent_of(idb, "c2"))
        self.assertTrue(ttns.is_child_of(idb, ida))
        self.assertTrue(ttns.is_parent_of(ida, idb))
        self.assertTrue(ttns.root_id == ida)

    def test_split_root_c1andphys_together_flipped(self):
        """
        In this test one node has the physical leg and the leg to c1 together.
        """
        ttns = random_small_ttns()
        tensor_a = crandn((3,5,2))
        tensor_b = crandn((6,3))
        ref_a = deepcopy(tensor_a)
        ref_b = deepcopy(tensor_b)
        ida = "a"
        idb = "b"
        legs_a = LegSpecification(None, ["c1"], [2], is_root=True)
        legs_b = LegSpecification(None, ["c2"], [])
        ttns.split_node_replace("root", tensor_b, tensor_a,
                                    idb, ida,
                                    legs_b, legs_a)
        # A child order = b, c1
        self.assertTrue(ida in ttns.nodes)
        self.assertTrue(idb in ttns.nodes)
        self.assertTrue(ida in ttns.tensors)
        self.assertTrue(idb in ttns.tensors)
        self.assertTrue(allclose(ref_a, ttns.tensors[ida]))
        self.assertTrue(allclose(ref_b.transpose(1,0), ttns.tensors[idb]))
        self.assertTrue(ttns.is_child_of("c1", ida))
        self.assertTrue(ttns.is_parent_of(ida, "c1"))
        self.assertTrue(ttns.is_child_of("c2", idb))
        self.assertTrue(ttns.is_parent_of(idb, "c2"))
        self.assertTrue(ttns.is_child_of(idb, ida))
        self.assertTrue(ttns.is_parent_of(ida, idb))
        self.assertTrue(ttns.root_id == ida)

    def test_split_root_c2andphys_together(self):
        """
        In this test one node has the physical leg and the leg to c2 together.
        """
        ttns = random_small_ttns()
        tensor_a = crandn((5,3))
        tensor_b = crandn((3,6,2))
        ref_a = deepcopy(tensor_a)
        ref_b = deepcopy(tensor_b)
        ida = "a"
        idb = "b"
        legs_a = LegSpecification(None, ["c1"], [], is_root=True)
        legs_b = LegSpecification(None, ["c2"], [2])
        ttns.split_node_replace("root", tensor_a, tensor_b,
                                    ida, idb,
                                    legs_a, legs_b)
        # A child order = b, c1
        self.assertTrue(ida in ttns.nodes)
        self.assertTrue(idb in ttns.nodes)
        self.assertTrue(ida in ttns.tensors)
        self.assertTrue(idb in ttns.tensors)
        self.assertTrue(allclose(ref_a.transpose(1,0), ttns.tensors[ida]))
        self.assertTrue(allclose(ref_b, ttns.tensors[idb]))
        self.assertTrue(ttns.is_child_of("c1", ida))
        self.assertTrue(ttns.is_parent_of(ida, "c1"))
        self.assertTrue(ttns.is_child_of("c2", idb))
        self.assertTrue(ttns.is_parent_of(idb, "c2"))
        self.assertTrue(ttns.is_child_of(idb, ida))
        self.assertTrue(ttns.is_parent_of(ida, idb))
        self.assertTrue(ttns.root_id == ida)

    def test_split_root_c2andphys_together_flipped(self):
        """
        In this test one node has the physical leg and the leg to c2 together.
        """
        ttns = random_small_ttns()
        tensor_a = crandn((3,5))
        tensor_b = crandn((6,2,3))
        ref_a = deepcopy(tensor_a)
        ref_b = deepcopy(tensor_b)
        ida = "a"
        idb = "b"
        legs_a = LegSpecification(None, ["c1"], [], is_root=True)
        legs_b = LegSpecification(None, ["c2"], [2])
        ttns.split_node_replace("root", tensor_b, tensor_a,
                                    idb, ida,
                                    legs_b, legs_a)
        # A child order = b, c1
        self.assertTrue(ida in ttns.nodes)
        self.assertTrue(idb in ttns.nodes)
        self.assertTrue(ida in ttns.tensors)
        self.assertTrue(idb in ttns.tensors)
        self.assertTrue(allclose(ref_a, ttns.tensors[ida]))
        self.assertTrue(allclose(ref_b.transpose(2,0,1), ttns.tensors[idb]))
        self.assertTrue(ttns.is_child_of("c1", ida))
        self.assertTrue(ttns.is_parent_of(ida, "c1"))
        self.assertTrue(ttns.is_child_of("c2", idb))
        self.assertTrue(ttns.is_parent_of(idb, "c2"))
        self.assertTrue(ttns.is_child_of(idb, ida))
        self.assertTrue(ttns.is_parent_of(ida, idb))
        self.assertTrue(ttns.root_id == ida)


if __name__ == "__main__":
    unittest.main()