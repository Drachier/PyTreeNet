import unittest

from copy import deepcopy, copy

from numpy import allclose, savez, load
from scipy.linalg import expm
import unittest.test

from pytreenet.time_evolution.bug import BUG
from pytreenet.time_evolution.time_evo_util.bug_util import (basis_change_tensor_id,
                                                            reverse_basis_change_tensor_id)
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.random.random_ttns import RandomTTNSMode
from pytreenet.time_evolution.time_evolution import time_evolve

from pytreenet.contractions.sandwich_caching import SandwichCache
from pytreenet.contractions.effective_hamiltonians import get_effective_single_site_hamiltonian
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict

def init_simple_bug() -> BUG:
    """
    Initialise a simple BUG instance.
    """
    ttns, ttno = small_ttns_and_ttno()
    time_step_size = 0.1
    final_time = 1.0
    bug = BUG(ttns, ttno,
                time_step_size, final_time,
                [])
    return bug

def init_complicated_bug() -> BUG:
    """
    Initialise a complicated BUG instance.
    """
    ttns, ttno = big_ttns_and_ttno(mode=RandomTTNSMode.DIFFVIRT)
    time_step_size = 0.1
    final_time = 1.0
    bug = BUG(ttns, ttno,
                time_step_size, final_time,
                [])
    return bug

class TestBugInitSimple(unittest.TestCase):

    def build_ref_cache(self, ref_ttns, ref_ttno) -> SandwichCache:
        """
        Build the correct cache to appear after initialisation.
        """
        ref_cache = SandwichCache(ref_ttns, ref_ttno)
        ref_cache.update_tree_cache("c1", "root")
        ref_cache.update_tree_cache("c2", "root")
        return ref_cache

    def test_init_not_orth(self):
        """
        Initialise the BUG class with a non-orthogonalised state.
        """
        ttns, ttno = small_ttns_and_ttno()
        ref_ttns = deepcopy(ttns)
        ref_ttno = deepcopy(ttno)
        bug = BUG(ttns, ttno, 0.1, 1.0, [])
        self.assertEqual(bug.hamiltonian, ref_ttno)
        ref_ttns.canonical_form("root")
        self.assertEqual(bug.state, ref_ttns)
        ref_cache = self.build_ref_cache(ref_ttns, ref_ttno)
        self.assertTrue(bug.tensor_cache.close_to(ref_cache))
        self.assertEqual(bug.initial_state, ref_ttns)
        self.assertEqual(bug.ttns_dict, {})
        self.assertEqual(bug.cache_dict, {})
        self.assertEqual(bug.basis_change_cache, PartialTreeCachDict())

    def test_init_wrong_orth(self):
        """
        Initialise the BUG class with a wrongly orthogonalised state.
        """
        ttns, ttno = small_ttns_and_ttno()
        ttns.canonical_form("c1")
        ref_ttns = deepcopy(ttns)
        ref_ttno = deepcopy(ttno)
        bug = BUG(ttns, ttno, 0.1, 1.0, [])
        self.assertEqual(bug.hamiltonian, ref_ttno)
        ref_ttns.move_orthogonalization_center("root")
        self.assertEqual(bug.state, ref_ttns)
        ref_cache = self.build_ref_cache(ref_ttns, ref_ttno)
        self.assertTrue(bug.tensor_cache.close_to(ref_cache))
        self.assertEqual(bug.initial_state, ref_ttns)
        self.assertEqual(bug.ttns_dict, {})
        self.assertEqual(bug.cache_dict, {})
        self.assertEqual(bug.basis_change_cache, PartialTreeCachDict())

    def test_init_orth(self):
        """
        Initialise the BUG class with a correctly orthogonalised state.
        """
        ttns, ttno = small_ttns_and_ttno()
        ttns.canonical_form("root")
        ref_ttns = deepcopy(ttns)
        ref_ttno = deepcopy(ttno)
        bug = BUG(ttns, ttno, 0.1, 1.0, [])
        self.assertEqual(bug.hamiltonian, ref_ttno)
        self.assertEqual(bug.state, ref_ttns)
        ref_cache = self.build_ref_cache(ref_ttns, ref_ttno)
        self.assertTrue(bug.tensor_cache.close_to(ref_cache))
        self.assertEqual(bug.initial_state, ref_ttns)
        self.assertEqual(bug.ttns_dict, {})
        self.assertEqual(bug.cache_dict, {})
        self.assertEqual(bug.basis_change_cache, PartialTreeCachDict())

class TestPrepareRootStateMethod(unittest.TestCase):
    """
    Test the prepare_root_state method of the BUG class.
    """

    def test_prepare_root_state_exception(self):
        """
        The state dictionary should be empty at the beginning.
        """
        bug = init_simple_bug()
        bug.ttns_dict = {"root": bug.state}
        self.assertRaises(AssertionError, bug.prepare_root_state)

    def test_prepare_root_state_simple(self):
        """
        Prepare the root state in a simple case.
        """
        bug = init_simple_bug()
        ref_state = deepcopy(bug.state)
        root_state = bug.prepare_root_state()
        self.assertEqual(root_state, ref_state)
        self.assertEqual(1, len(bug.ttns_dict))
        root_id = ref_state.root_id
        self.assertIn(root_id, bug.ttns_dict)
        self.assertEqual(id(root_state), id(bug.ttns_dict[root_id]))

    def test_prepare_root_state_complicated(self):
        """
        Prepare the root state in a more compicated case.
        """
        bug = init_complicated_bug()
        ref_state = deepcopy(bug.state)
        root_state = bug.prepare_root_state()
        self.assertEqual(root_state, ref_state)
        self.assertEqual(1, len(bug.ttns_dict))
        root_id = ref_state.root_id
        self.assertIn(root_id, bug.ttns_dict)
        self.assertEqual(id(root_state), id(bug.ttns_dict[root_id]))

class TestPrepareRootCacheMethod(unittest.TestCase):
    """
    Test the prepare_root_cache method of the BUG class.
    """

    def test_prepare_root_cache_exception(self):
        """
        The cache dictionary should be empty at the beginning.
        """
        bug = init_simple_bug()
        bug.cache_dict = {"root": SandwichCache(bug.state, bug.hamiltonian)}
        self.assertRaises(AssertionError, bug.prepare_root_cache)

    def test_prepare_root_cache_simple(self):
        """
        Test the prepare_root_cache method in a simple case.
        """
        bug = init_simple_bug()
        ref_cache = SandwichCache.init_cache_but_one(deepcopy(bug.state),
                                                     deepcopy(bug.hamiltonian),
                                                     bug.state.root_id)
        root_cache = bug.prepare_root_cache()
        self.assertTrue(root_cache.close_to(ref_cache))
        self.assertEqual(1, len(bug.cache_dict))
        root_id = bug.state.root_id
        self.assertIn(root_id, bug.cache_dict)
        self.assertEqual(id(root_cache), id(bug.cache_dict[root_id]))

    def test_prepare_root_cache_complicated(self):
        """
        Test the prepare_root_cache method in a complicated case.
        """
        bug = init_complicated_bug()
        ref_cache = SandwichCache.init_cache_but_one(deepcopy(bug.state),
                                                     deepcopy(bug.hamiltonian),
                                                     bug.state.root_id)
        root_cache = bug.prepare_root_cache()
        self.assertTrue(root_cache.close_to(ref_cache))
        self.assertEqual(1, len(bug.cache_dict))
        root_id = bug.state.root_id
        self.assertIn(root_id, bug.cache_dict)
        self.assertEqual(id(root_cache), id(bug.cache_dict[root_id]))

class TestPrepareRootForChildUpdateMethhod(unittest.TestCase):
    """
    Test the prepare_root_for_child_update method of the BUG class.

    This means, the state and the cache should be prepared and added to the
    respective dictionaries.
    """

    def test_wrong_orth_center(self):
        """
        There should be an Exception, if the orthogonality center is not the root.
        """
        bug = init_simple_bug()
        bug.state.move_orthogonalization_center("c1")
        self.assertRaises(AssertionError, bug.prepare_root_for_child_update)

    def test_simple(self):
        """
        Test the prepare_root_for_child_update method in a simple case.
        """
        bug = init_simple_bug()
        ref_state_bug = deepcopy(bug)
        ref_state_bug.prepare_root_state()
        ref_cache_bug = deepcopy(bug)
        ref_cache_bug.prepare_root_cache()
        bug.prepare_root_for_child_update()
        # Check State Dict
        self.assertEqual(ref_state_bug.ttns_dict,
                         bug.ttns_dict)
        # Check Cache Dict
        self.assertTrue(ref_cache_bug.tensor_cache.close_to(bug.tensor_cache))

    def test_complicated(self):
        """
        Test the prepare_root_for_child_update method in a complicated case.
        """
        bug = init_complicated_bug()
        ref_state_bug = deepcopy(bug)
        ref_state_bug.prepare_root_state()
        ref_cache_bug = deepcopy(bug)
        ref_cache_bug.prepare_root_cache()
        bug.prepare_root_for_child_update()
        # Check State Dict
        self.assertEqual(ref_state_bug.ttns_dict,
                         bug.ttns_dict)
        # Check Cache Dict
        self.assertTrue(ref_cache_bug.tensor_cache.close_to(bug.tensor_cache))

class TestGetChildrenIdsToUpdateMethod(unittest.TestCase):
    """
    This method should return an frozenset of the children ids.
    """

    def test_simple_root(self):
        """
        Tests the method for the root of the simple case.
        """
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        root_id = bug.state.root_id
        children_ids = bug.get_children_ids_to_update(root_id)
        self.assertEqual(children_ids, frozenset({"c1", "c2"}))

    def test_complicated_root(self):
        """
        Tests the method for the root of the complicated case.
        """
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        root_id = bug.state.root_id
        children_ids = bug.get_children_ids_to_update(root_id)
        self.assertEqual(children_ids, frozenset({"site1", "site6"}))

    def test_simple_c1(self):
        """
        Test the method for the node c1 of the simple case.
        """
        node_id = "c1"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        children_ids = bug.get_children_ids_to_update(node_id)
        self.assertEqual(children_ids, frozenset())

    def test_simple_c2(self):
        """
        Test the method for the node c2 of the simple case.
        """
        node_id = "c2"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        children_ids = bug.get_children_ids_to_update(node_id)
        self.assertEqual(children_ids, frozenset())

    def test_complicated_site1(self):
        """
        Test the method for the node site1 of the complicated case.
        """
        node_id = "site1"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        children_ids = bug.get_children_ids_to_update(node_id)
        self.assertEqual(children_ids, frozenset({"site2", "site3"}))

    def test_complicated_site6(self):
        """
        Test the method for the node site6 of the complicated case.
        """
        node_id = "site6"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        children_ids = bug.get_children_ids_to_update(node_id)
        self.assertEqual(children_ids, frozenset({"site7"}))

    def test_complicated_site2(self):
        """
        Test the method for the node site2 of the complicated case.
        """
        node_id = "site2"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        children_ids = bug.get_children_ids_to_update(node_id)
        self.assertEqual(children_ids, frozenset())

    def test_complicated_site7(self):
        """
        Test the method for the node site7 of the complicated case.
        """
        node_id = "site7"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site6")
        bug.create_ttns_and_cache(node_id)
        children_ids = bug.get_children_ids_to_update(node_id)
        self.assertEqual(children_ids, frozenset())

    def test_complicated_site3(self):
        """
        Test the method for the node site3 of the complicated case.
        """
        node_id = "site3"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        children_ids = bug.get_children_ids_to_update(node_id)
        self.assertEqual(children_ids, frozenset({"site4", "site5"}))

    def test_complicated_site4(self):
        """
        Test the method for the node site4 of the complicated case.
        """
        node_id = "site4"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        children_ids = bug.get_children_ids_to_update(node_id)
        self.assertEqual(children_ids, frozenset())

    def test_complicated_site5(self):
        """
        Test the method for the node site5 of the complicated case.
        """
        node_id = "site5"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        children_ids = bug.get_children_ids_to_update(node_id)
        self.assertEqual(children_ids, frozenset())


class TestPrepareNodeStateMethod(unittest.TestCase):
    """
    Tests the prepare_node_state method of the BUG class.
    """

    def test_simple_c1(self):
        """
        Test the method for the node c1 of the simple case.
        """
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        # Prepare reference
        ref_state = deepcopy(bug.ttns_dict[bug.state.root_id])
        ref_state.move_orthogonalization_center("c1")
        found_state = bug.prepare_node_state("c1")
        self.assertEqual(ref_state, found_state)
        expected_length = 2
        self.assertEqual(expected_length, len(bug.ttns_dict))
        self.assertIn("c1", bug.ttns_dict)
        self.assertEqual(id(found_state), id(bug.ttns_dict["c1"]))

    def test_simple_c2(self):
        """
        Test the method for the node c2 of the simple case.
        """
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        # Prepare reference
        ref_state = deepcopy(bug.ttns_dict[bug.state.root_id])
        ref_state.move_orthogonalization_center("c2")
        found_state = bug.prepare_node_state("c2")
        self.assertEqual(ref_state, found_state)
        expected_length = 2
        self.assertEqual(expected_length, len(bug.ttns_dict))
        self.assertIn("c2", bug.ttns_dict)
        self.assertEqual(id(found_state), id(bug.ttns_dict["c2"]))

    def test_complicated_site1(self):
        """
        Test the method for the node site1 of the complicated case.
        """
        node_id = "site1"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        # Prepare reference
        ref_state = deepcopy(bug.ttns_dict[bug.state.root_id])
        ref_state.move_orthogonalization_center(node_id)
        found_state = bug.prepare_node_state(node_id)
        self.assertEqual(ref_state, found_state)
        expected_length = 2
        self.assertEqual(expected_length, len(bug.ttns_dict))
        self.assertIn(node_id, bug.ttns_dict)
        self.assertEqual(id(found_state), id(bug.ttns_dict[node_id]))

    def test_complicated_site6(self):
        """
        Test the method for the node site6 of the complicated case.
        """
        node_id = "site6"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        # Prepare reference
        ref_state = deepcopy(bug.ttns_dict[bug.state.root_id])
        ref_state.move_orthogonalization_center(node_id)
        found_state = bug.prepare_node_state(node_id)
        self.assertEqual(ref_state, found_state)
        expected_length = 2
        self.assertEqual(expected_length, len(bug.ttns_dict))
        self.assertIn(node_id, bug.ttns_dict)
        self.assertEqual(id(found_state), id(bug.ttns_dict[node_id]))

    def test_complicated_site2(self):
        """
        Test the method for the node site2 of the complicated case.
        """
        node_id = "site2"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.prepare_node_state("site1")
        # Prepare reference
        ref_state = deepcopy(bug.ttns_dict[bug.state.root_id])
        ref_state.move_orthogonalization_center(node_id)
        found_state = bug.prepare_node_state(node_id)
        self.assertEqual(ref_state, found_state)
        expected_length = 3
        self.assertEqual(expected_length, len(bug.ttns_dict))
        self.assertIn(node_id, bug.ttns_dict)
        self.assertEqual(id(found_state), id(bug.ttns_dict[node_id]))

    def test_complicated_site7(self):
        """
        Test the method for the node site7 of the complicated case.
        """
        node_id = "site7"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.prepare_node_state("site6")
        # Prepare reference
        ref_state = deepcopy(bug.ttns_dict[bug.state.root_id])
        ref_state.move_orthogonalization_center(node_id)
        found_state = bug.prepare_node_state(node_id)
        self.assertEqual(ref_state, found_state)
        expected_length = 3
        self.assertEqual(expected_length, len(bug.ttns_dict))
        self.assertIn(node_id, bug.ttns_dict)
        self.assertEqual(id(found_state), id(bug.ttns_dict[node_id]))

    def test_complicated_site3(self):
        """
        Test the method for the node site3 of the complicated case.
        """
        node_id = "site3"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.prepare_node_state("site1")
        # Prepare reference
        ref_state = deepcopy(bug.ttns_dict[bug.state.root_id])
        ref_state.move_orthogonalization_center(node_id)
        found_state = bug.prepare_node_state(node_id)
        self.assertEqual(ref_state, found_state)
        expected_length = 3
        self.assertEqual(expected_length, len(bug.ttns_dict))
        self.assertIn(node_id, bug.ttns_dict)
        self.assertEqual(id(found_state), id(bug.ttns_dict[node_id]))

    def test_complicated_site4(self):
        """
        Test the method for the node site4 of the complicated case.
        """
        node_id = "site4"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.prepare_node_state("site1")
        bug.prepare_node_state("site3")
        # Prepare reference
        ref_state = deepcopy(bug.ttns_dict[bug.state.root_id])
        ref_state.move_orthogonalization_center(node_id)
        found_state = bug.prepare_node_state(node_id)
        self.assertEqual(ref_state, found_state)
        expected_length = 4
        self.assertEqual(expected_length, len(bug.ttns_dict))
        self.assertIn(node_id, bug.ttns_dict)
        self.assertEqual(id(found_state), id(bug.ttns_dict[node_id]))

    def test_complicated_site5(self):
        """
        Test the method for the node site5 of the complicated case.
        """
        node_id = "site5"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.prepare_node_state("site1")
        bug.prepare_node_state("site3")
        # Prepare reference
        ref_state = deepcopy(bug.ttns_dict[bug.state.root_id])
        ref_state.move_orthogonalization_center(node_id)
        found_state = bug.prepare_node_state(node_id)
        self.assertEqual(ref_state, found_state)
        expected_length = 4
        self.assertEqual(expected_length, len(bug.ttns_dict))
        self.assertIn(node_id, bug.ttns_dict)
        self.assertEqual(id(found_state), id(bug.ttns_dict[node_id]))

class TestPrepareNodeCacheMethod(unittest.TestCase):
    """
    Tests the prepare_node_cache method of the BUG class.
    """

    def test_simple_c1(self):
        """
        Tests the method for the node c1 of the simple case.
        """
        node_id = "c1"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_state = ref_bug.prepare_node_state(node_id)
        ref_cache = SandwichCache.init_cache_but_one(deepcopy(ref_state),
                                                        deepcopy(ref_bug.hamiltonian),
                                                        node_id)
        found_cache = bug.prepare_node_cache(node_id)
        self.assertTrue(found_cache.close_to(ref_cache))
        expected_length = 2
        self.assertEqual(expected_length, len(bug.cache_dict))
        self.assertIn(node_id, bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))

    def test_simple_c2(self):
        """
        Tests the method for the node c2 of the simple case.
        """
        node_id = "c2"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_state = ref_bug.prepare_node_state(node_id)
        ref_cache = SandwichCache.init_cache_but_one(deepcopy(ref_state),
                                                        deepcopy(ref_bug.hamiltonian),
                                                        node_id)
        found_cache = bug.prepare_node_cache(node_id)
        self.assertTrue(found_cache.close_to(ref_cache))
        expected_length = 2
        self.assertEqual(expected_length, len(bug.cache_dict))
        self.assertIn(node_id, bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))

    def test_complicated_site1(self):
        """
        Tests the method for the node site1 of the complicated case.
        """
        node_id = "site1"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_state = ref_bug.prepare_node_state(node_id)
        ref_cache = SandwichCache.init_cache_but_one(deepcopy(ref_state),
                                                        deepcopy(ref_bug.hamiltonian),
                                                        node_id)
        found_cache = bug.prepare_node_cache(node_id)
        self.assertTrue(found_cache.close_to(ref_cache))
        expected_length = 2
        self.assertEqual(expected_length, len(bug.cache_dict))
        self.assertIn(node_id, bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))

    def test_complicated_site6(self):
        """
        Tests the method for the node site6 of the complicated case.
        """
        node_id = "site6"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_state = ref_bug.prepare_node_state(node_id)
        ref_cache = SandwichCache.init_cache_but_one(deepcopy(ref_state),
                                                        deepcopy(ref_bug.hamiltonian),
                                                        node_id)
        found_cache = bug.prepare_node_cache(node_id)
        self.assertTrue(found_cache.close_to(ref_cache))
        expected_length = 2
        self.assertEqual(expected_length, len(bug.cache_dict))
        self.assertIn(node_id, bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))

    def test_complicated_site2(self):
        """
        Tests the method for the node site2 of the complicated case.
        """
        node_id = "site2"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_state = ref_bug.prepare_node_state(node_id)
        ref_cache = SandwichCache.init_cache_but_one(deepcopy(ref_state),
                                                        deepcopy(ref_bug.hamiltonian),
                                                        node_id)
        found_cache = bug.prepare_node_cache(node_id)
        self.assertTrue(found_cache.close_to(ref_cache))
        expected_length = 3
        self.assertEqual(expected_length, len(bug.cache_dict))
        self.assertIn(node_id, bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))

    def test_complicated_site7(self):
        """
        Tests the method for the node site7 of the complicated case.
        """
        node_id = "site7"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site6")
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_state = ref_bug.prepare_node_state(node_id)
        ref_cache = SandwichCache.init_cache_but_one(deepcopy(ref_state),
                                                        deepcopy(ref_bug.hamiltonian),
                                                        node_id)
        found_cache = bug.prepare_node_cache(node_id)
        self.assertTrue(found_cache.close_to(ref_cache))
        expected_length = 3
        self.assertEqual(expected_length, len(bug.cache_dict))
        self.assertIn(node_id, bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))

    def test_complicated_site3(self):
        """
        Tests the method for the node site3 of the complicated case.
        """
        node_id = "site3"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_state = ref_bug.prepare_node_state(node_id)
        ref_cache = SandwichCache.init_cache_but_one(deepcopy(ref_state),
                                                        deepcopy(ref_bug.hamiltonian),
                                                        node_id)
        found_cache = bug.prepare_node_cache(node_id)
        self.assertTrue(found_cache.close_to(ref_cache))
        expected_length = 3
        self.assertEqual(expected_length, len(bug.cache_dict))
        self.assertIn(node_id, bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))

    def test_complicated_site4(self):
        """
        Tests the method for the node site4 of the complicated case.
        """
        node_id = "site4"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_state = ref_bug.prepare_node_state(node_id)
        ref_cache = SandwichCache.init_cache_but_one(deepcopy(ref_state),
                                                        deepcopy(ref_bug.hamiltonian),
                                                        node_id)
        found_cache = bug.prepare_node_cache(node_id)
        self.assertTrue(found_cache.close_to(ref_cache))
        expected_length = 4
        self.assertEqual(expected_length, len(bug.cache_dict))
        self.assertIn(node_id, bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))

    def test_complicated_site5(self):
        """
        Tests the method for the node site5 of the complicated case.
        """
        node_id = "site5"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_state = ref_bug.prepare_node_state(node_id)
        ref_cache = SandwichCache.init_cache_but_one(deepcopy(ref_state),
                                                        deepcopy(ref_bug.hamiltonian),
                                                        node_id)
        found_cache = bug.prepare_node_cache(node_id)
        self.assertTrue(found_cache.close_to(ref_cache))
        expected_length = 4
        self.assertEqual(expected_length, len(bug.cache_dict))
        self.assertIn(node_id, bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))

class TestCreateTTNSAndCacheMethod(unittest.TestCase):
    """
    Tests the create_ttns_and_cache method of the BUG class.

    It should prepare the cache and ttns for the current node.

    Lower nodes, require their ancestors to be prepared already.
    """

    def test_simple_c1(self):
        """
        Test the method for the node c1 of the simple case.
        """
        node_id = "c1"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        # References
        ref_state_bug = deepcopy(bug)
        ref_state = ref_state_bug.prepare_node_state(node_id)
        ref_cache_bug = deepcopy(bug)
        ref_cache = ref_cache_bug.prepare_node_cache(node_id)
        # Test
        found_ttns, found_cache = bug.create_ttns_and_cache(node_id)
        self.assertEqual(ref_state, found_ttns)
        self.assertTrue(found_cache.close_to(ref_cache))
        self.assertEqual(ref_state, bug.ttns_dict[node_id])
        self.assertTrue(found_cache.close_to(bug.cache_dict[node_id]))

    def test_simple_c2(self):
        """
        Test the method for the node c2 of the simple case.
        """
        node_id = "c2"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        # References
        ref_state_bug = deepcopy(bug)
        ref_state = ref_state_bug.prepare_node_state(node_id)
        ref_cache_bug = deepcopy(bug)
        ref_cache = ref_cache_bug.prepare_node_cache(node_id)
        # Test
        found_ttns, found_cache = bug.create_ttns_and_cache(node_id)
        self.assertEqual(ref_state, found_ttns)
        self.assertTrue(found_cache.close_to(ref_cache))
        self.assertEqual(ref_state, bug.ttns_dict[node_id])
        self.assertTrue(found_cache.close_to(bug.cache_dict[node_id]))

    def test_complicated_site1(self):
        """
        Test the method for the node site1 of the complicated case.
        """
        node_id = "site1"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        # References
        ref_state_bug = deepcopy(bug)
        ref_state = ref_state_bug.prepare_node_state(node_id)
        ref_cache_bug = deepcopy(bug)
        ref_cache = ref_cache_bug.prepare_node_cache(node_id)
        # Test
        found_ttns, found_cache = bug.create_ttns_and_cache(node_id)
        self.assertEqual(ref_state, found_ttns)
        self.assertTrue(found_cache.close_to(ref_cache))
        self.assertEqual(ref_state, bug.ttns_dict[node_id])
        self.assertTrue(found_cache.close_to(bug.cache_dict[node_id]))

    def test_complicated_site6(self):
        """
        Test the method for the node site6 of the complicated case.
        """
        node_id = "site6"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        # References
        ref_state_bug = deepcopy(bug)
        ref_state = ref_state_bug.prepare_node_state(node_id)
        ref_cache_bug = deepcopy(bug)
        ref_cache = ref_cache_bug.prepare_node_cache(node_id)
        # Test
        found_ttns, found_cache = bug.create_ttns_and_cache(node_id)
        self.assertEqual(ref_state, found_ttns)
        self.assertTrue(found_cache.close_to(ref_cache))
        self.assertEqual(ref_state, bug.ttns_dict[node_id])
        self.assertTrue(found_cache.close_to(bug.cache_dict[node_id]))

    def test_complicated_site2(self):
        """
        Test the method for the node site2 of the complicated case.
        """
        node_id = "site2"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        # References
        ref_state_bug = deepcopy(bug)
        ref_state = ref_state_bug.prepare_node_state(node_id)
        ref_cache_bug = deepcopy(bug)
        ref_cache = ref_cache_bug.prepare_node_cache(node_id)
        # Test
        found_ttns, found_cache = bug.create_ttns_and_cache(node_id)
        self.assertEqual(ref_state, found_ttns)
        self.assertTrue(found_cache.close_to(ref_cache))
        self.assertEqual(ref_state, bug.ttns_dict[node_id])
        self.assertTrue(found_cache.close_to(bug.cache_dict[node_id]))

    def test_complicated_site7(self):
        """
        Test the method for the node site7 of the complicated case.
        """
        node_id = "site7"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site6")
        # References
        ref_state_bug = deepcopy(bug)
        ref_state = ref_state_bug.prepare_node_state(node_id)
        ref_cache_bug = deepcopy(bug)
        ref_cache = ref_cache_bug.prepare_node_cache(node_id)
        # Test
        found_ttns, found_cache = bug.create_ttns_and_cache(node_id)
        self.assertEqual(ref_state, found_ttns)
        self.assertTrue(found_cache.close_to(ref_cache))
        self.assertEqual(ref_state, bug.ttns_dict[node_id])
        self.assertTrue(found_cache.close_to(bug.cache_dict[node_id]))

    def test_complicated_site3(self):
        """
        Test the method for the node site3 of the complicated case.
        """
        node_id = "site3"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        # References
        ref_state_bug = deepcopy(bug)
        ref_state = ref_state_bug.prepare_node_state(node_id)
        ref_cache_bug = deepcopy(bug)
        ref_cache = ref_cache_bug.prepare_node_cache(node_id)
        # Test
        found_ttns, found_cache = bug.create_ttns_and_cache(node_id)
        self.assertEqual(ref_state, found_ttns)
        self.assertTrue(found_cache.close_to(ref_cache))
        self.assertEqual(ref_state, bug.ttns_dict[node_id])
        self.assertTrue(found_cache.close_to(bug.cache_dict[node_id]))

    def test_complicated_site4(self):
        """
        Test the method for the node site4 of the complicated case.
        """
        node_id = "site4"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        # References
        ref_state_bug = deepcopy(bug)
        ref_state = ref_state_bug.prepare_node_state(node_id)
        ref_cache_bug = deepcopy(bug)
        ref_cache = ref_cache_bug.prepare_node_cache(node_id)
        # Test
        found_ttns, found_cache = bug.create_ttns_and_cache(node_id)
        self.assertEqual(ref_state, found_ttns)
        self.assertTrue(found_cache.close_to(ref_cache))
        self.assertEqual(ref_state, bug.ttns_dict[node_id])
        self.assertTrue(found_cache.close_to(bug.cache_dict[node_id]))

    def test_complicated_site5(self):
        """
        Test the method for the node site5 of the complicated case.
        """
        node_id = "site5"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        # References
        ref_state_bug = deepcopy(bug)
        ref_state = ref_state_bug.prepare_node_state(node_id)
        ref_cache_bug = deepcopy(bug)
        ref_cache = ref_cache_bug.prepare_node_cache(node_id)
        # Test
        found_ttns, found_cache = bug.create_ttns_and_cache(node_id)
        self.assertEqual(ref_state, found_ttns)
        self.assertTrue(found_cache.close_to(ref_cache))
        self.assertEqual(ref_state, bug.ttns_dict[node_id])
        self.assertTrue(found_cache.close_to(bug.cache_dict[node_id]))

class TestUpdateChildren(unittest.TestCase):
    """
    Test, if the children are updated correctly.
    """

    # Nothing should happen for the leafs.
    def test_simple_c1(self):
        """
        Test the method for the node c1 of the simple case.
        """
        node_id = "c1"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        ref_bug = deepcopy(bug)
        bug.update_children(node_id)
        self.assertTrue(bug.close_to(ref_bug))

    def test_simple_c2(self):
        """
        Test the method for the node c2 of the simple case.
        """
        node_id = "c2"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        ref_bug = deepcopy(bug)
        bug.update_children(node_id)
        self.assertTrue(bug.close_to(ref_bug))

    def test_complicated_site2(self):
        """
        Test the method for the node site2 of the complicated case.
        """
        node_id = "site2"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        ref_bug = deepcopy(bug)
        bug.update_children(node_id)
        self.assertTrue(bug.close_to(ref_bug))

    def test_complicated_site7(self):
        """
        Test the method for the node site7 of the complicated case.
        """
        node_id = "site7"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site6")
        bug.create_ttns_and_cache(node_id)
        ref_bug = deepcopy(bug)
        bug.update_children(node_id)
        self.assertTrue(bug.close_to(ref_bug))

    def test_complicated_site4(self):
        """
        Test the method for the node site4 of the complicated case.
        """
        node_id = "site4"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        ref_bug = deepcopy(bug)
        bug.update_children(node_id)
        self.assertTrue(bug.close_to(ref_bug))

    def test_complicated_site5(self):
        """
        Test the method for the node site5 of the complicated case.
        """
        node_id = "site5"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        ref_bug = deepcopy(bug)
        bug.update_children(node_id)
        self.assertTrue(bug.close_to(ref_bug))

    # TODO: Write test for remaining nodes, once all required methods are adapted

class TestPrepareStateForUpdateMethod(unittest.TestCase):
    """
    Tests the preparing of the main state for the update of a given node.
    """

    # For the leafs, their orthogonality center is pulled into the main state
    def test_simple_c1(self):
        """
        Test the method for the node c1 of the simple case.
        """
        node_id = "c1"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_node = deepcopy(bug.state.nodes[node_id])
        ref_tensor = deepcopy(bug.ttns_dict[node_id].tensors[node_id])
        ref_state = deepcopy(bug.state)
        # Test
        bug.prepare_state_for_update(node_id)
        found_node, found_tensor = bug.state[node_id]
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        for key in bug.state.nodes:
            print(key)
            print(bug.state.nodes[key])
            print(ref_state.nodes[key])
            print(allclose(bug.state.tensors[key], ref_state.tensors[key]))
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(node_id,bug.state))



if __name__ == "__main__":
    unittest.main()
