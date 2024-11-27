from unittest import TestCase, main as unit_main

from copy import deepcopy, copy

from numpy import allclose, eye, tensordot
from scipy.linalg import expm

from pytreenet.time_evolution.bug import BUG
from pytreenet.time_evolution.time_evo_util.bug_util import (basis_change_tensor_id)
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.random.random_ttns import RandomTTNSMode
from pytreenet.random.random_matrices import crandn_like, crandn
from pytreenet.core.node import Node
from pytreenet.contractions.state_operator_contraction import contract_leaf
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

def init_complicated_bug(mode=RandomTTNSMode.DIFFVIRT) -> BUG:
    """
    Initialise a complicated BUG instance.
    """
    ttns, ttno = big_ttns_and_ttno(mode=mode)
    time_step_size = 0.1
    final_time = 1.0
    bug = BUG(ttns, ttno,
                time_step_size, final_time,
                [])
    return bug

class TestBugInitSimple(TestCase):

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

class TestPrepareRootStateMethod(TestCase):
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

class TestPrepareRootCacheMethod(TestCase):
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

class TestPrepareRootForChildUpdateMethhod(TestCase):
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

class TestGetChildrenIdsToUpdateMethod(TestCase):
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


class TestPrepareNodeStateMethod(TestCase):
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

class TestPrepareNodeCacheMethod(TestCase):
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

class TestCreateTTNSAndCacheMethod(TestCase):
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

class TestUpdateChildren(TestCase):
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

    # Test non-root nodes
    def test_site6_complicated(self):
        """
        Test the method for the node site6 of the complicated case.
        """
        node_id = "site6"
        child_id = "site7"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        ref_bug = deepcopy(bug)
        bug.update_children(node_id)
        # Check basis change tensor
        bc_id = basis_change_tensor_id(child_id)
        self.assertIn(bc_id, bug.state)
        bc_tensor = bug.state[bc_id][1] # Should be proportional to isometry
        dim = 2
        self.assertTrue(allclose(dim, tensordot(bc_tensor.conj(), bc_tensor,
                                              axes=([0,1],[1,0]))))
        # Check the new basis tensor
        basis_id = child_id
        self.assertIn(basis_id, bug.state)
        nb_tensor = bug.state[basis_id][1] # Should be different and orthogonal
        self.assertNotEqual(id(ref_bug.state[basis_id][1]),
                            id(nb_tensor))
        identity = eye(dim)
        self.assertTrue(allclose(identity,
                                 tensordot(nb_tensor.conj(), nb_tensor,
                                           axes=([1],[1]))))
        # Check basis change cache
        self.assertTrue(bug.basis_change_cache.contains(child_id, node_id))
        self.assertTrue(allclose(bc_tensor,
                                 bug.basis_change_cache.get_entry(child_id, node_id)))
        # Check the tensor cache
        self.assertTrue(bug.tensor_cache.contains(child_id,node_id))
        ref_tensor = tensordot(nb_tensor, bug.hamiltonian[child_id][1],
                               axes=([1],[2]))
        ref_tensor = tensordot(ref_tensor, nb_tensor.conj(),
                               axes=([2],[1]))
        self.assertTrue(allclose(ref_tensor,
                                 bug.tensor_cache.get_entry(child_id,node_id)))

    def test_site3_complicated(self):
        """
        Test the method for the node site3 of the complicated case.
        """
        node_id = "site3"
        children_ids = {"site4", "site5"}
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        ref_bug = deepcopy(bug)
        bug.update_children(node_id)
        for child_id in children_ids:
            # Check basis change tensor
            bc_id = basis_change_tensor_id(child_id)
            self.assertIn(bc_id, bug.state)
            bc_tensor = bug.state[bc_id][1] # Should be proportional to isometry
            dim = 2
            self.assertTrue(allclose(dim, tensordot(bc_tensor.conj(), bc_tensor,
                                                axes=([0,1],[1,0]))))
            # Check the new basis tensor
            basis_id = child_id
            self.assertIn(basis_id, bug.state)
            nb_tensor = bug.state[basis_id][1] # Should be different and orthogonal
            self.assertNotEqual(id(ref_bug.state[basis_id][1]),
                                id(nb_tensor))
            identity = eye(dim)
            self.assertTrue(allclose(identity,
                                    tensordot(nb_tensor.conj(), nb_tensor,
                                            axes=([1],[1]))))
            # Check basis change cache
            self.assertTrue(bug.basis_change_cache.contains(child_id, node_id))
            self.assertTrue(allclose(bc_tensor,
                                    bug.basis_change_cache.get_entry(child_id, node_id)))
            # Check the tensor cache
            self.assertTrue(bug.tensor_cache.contains(child_id,node_id))
            ref_tensor = tensordot(nb_tensor, bug.hamiltonian[child_id][1],
                                axes=([1],[2]))
            ref_tensor = tensordot(ref_tensor, nb_tensor.conj(),
                                axes=([2],[1]))
            self.assertTrue(allclose(ref_tensor,
                                    bug.tensor_cache.get_entry(child_id,node_id)))

    # TODO: Write test for remaining nodes (root, 0, 1), once all required methods are adapted

class TestPrepareStateForUpdateMethod(TestCase):
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
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,bug.state))

    def test_simple_c2(self):
        """
        Test the method for the node c2 of the simple case.
        """
        node_id = "c2"
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
        # Other nodes haven't changed
        for key in bug.state.nodes:
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,bug.state))

    def test_complicated_site2(self):
        """
        Test the method for the node site2 of the complicated case.
        """
        node_id = "site2"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
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
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,bug.state))

    def test_complicated_site7(self):
        """
        Test the method for the node site7 of the complicated case.
        """
        node_id = "site7"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site6")
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
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,bug.state))

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
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,bug.state))

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
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,bug.state))

        # Test non-root nodes
    def test_site6_complicated(self):
        """
        Test the method for the node site6 of the complicated case.
        """
        node_id = "site6"
        parent_id = "site0"
        child_id = "site7"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        # Prepare reference
        ref_state_tensor = deepcopy(bug.ttns_dict[node_id].tensors[node_id])
        ref_bc_tensor = deepcopy(bug.state[basis_change_tensor_id("site7")][1])
        ref_tensor = tensordot(ref_state_tensor, ref_bc_tensor,
                               axes=([1],[0])).transpose(0,2,1)
        ref_node = Node(identifier=node_id, tensor=ref_tensor)
        ref_node.open_leg_to_parent(parent_id,0)
        ref_node.open_leg_to_child(child_id,1)
        ref_state = deepcopy(bug.state)
        # Test
        bug.prepare_state_for_update(node_id)
        found_node, found_tensor = bug.state[node_id]
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        for key in bug.state.nodes:
            if key != node_id and key != child_id:
                print(key)
                print(bug.state.nodes[key])
                print(ref_state.nodes[key])
                self.assertTrue(ref_state.nodes_equal(key,bug.state))
        self.assertTrue(allclose(ref_state.tensors[child_id],
                                 bug.state.tensors[child_id]))

    def test_site3_complicated(self):
        """
        Test the method for the node site3 of the complicated case.
        """
        node_id = "site3"
        parent_id = "site1"
        children_ids = {"site4", "site5"}
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        # Prepare reference
        ref_state_tensor = deepcopy(bug.ttns_dict[node_id].tensors[node_id])
        ref_bc_tensor4 = deepcopy(bug.state[basis_change_tensor_id("site4")][1])
        ref_bc_tensor5 = deepcopy(bug.state[basis_change_tensor_id("site5")][1])
        ref_tensor = tensordot(ref_state_tensor, ref_bc_tensor4,
                                axes=([1],[0]))
        ref_tensor = tensordot(ref_tensor, ref_bc_tensor5,
                                axes=([1],[0]))
        ref_tensor = ref_tensor.transpose(0,2,3,1)
        ref_node = Node(identifier=node_id, tensor=ref_tensor)
        ref_node.open_leg_to_parent(parent_id,0)
        ref_node.open_legs_to_children({"site4":1, "site5":2})
        ref_state = deepcopy(bug.state)
        # Test
        bug.prepare_state_for_update(node_id)
        found_node, found_tensor = bug.state[node_id]
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        for key in bug.state.nodes:
            if key != node_id and key not in children_ids:
                self.assertTrue(ref_state.nodes_equal(key,bug.state))

        # TODO: When write tests for the other nodes, once all required methods are refactored

class TestPrepareCacheForUpdate(TestCase):
    """
    Tests if the cache of the node 
    """

    # Test for leafs
    # The cache should not change for leafs.
    def test_simple_c1(self):
        """
        Tests the function for the node c1 in the simple tree.
        """
        node_id = "c1"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        _, found_cache = bug.prepare_cache_for_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_simple_c2(self):
        """
        Tests the function for the node c2 in the simple tree.
        """
        node_id = "c2"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        _, found_cache = bug.prepare_cache_for_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_complicated_site2(self):
        """
        Tests the function for the node site2 in the complicated tree.
        """
        node_id = "site2"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        _, found_cache = bug.prepare_cache_for_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_complicated_site7(self):
        """
        Tests the function for the node site7 in the complicated tree.
        """
        node_id = "site7"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site6")
        bug.create_ttns_and_cache(node_id)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        _, found_cache = bug.prepare_cache_for_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_complicated_site4(self):
        """
        Tests the function for the node site4 in the complicated tree.
        """
        node_id = "site4"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        _, found_cache = bug.prepare_cache_for_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))

    def test_complicated_site5(self):
        """
        Tests the function for the node site4 in the complicated tree.
        """
        node_id = "site5"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        _, found_cache = bug.prepare_cache_for_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))

    # Test non-root nodes
    def test_site6_complicated(self):
        """
        Test the method for the node site6 of the complicated case.
        """
        node_id = "site6"
        child_id = "site7"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_bug.prepare_state_for_update(node_id)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        ref_cache.add_entry(child_id, node_id,
                            ref_bug.tensor_cache.get_entry(child_id, node_id))
        # Test
        found_state, found_cache = bug.prepare_cache_for_update(node_id)
        self.assertTrue(found_cache.close_to(ref_cache))
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertEqual(ref_bug.state, found_state)
        self.assertEqual(id(found_state), id(bug.state))

    def test_site3_complicated(self):
        """
        Test the method for the node site3 of the complicated case.
        """
        node_id = "site3"
        children_ids = {"site4", "site5"}
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_bug.prepare_state_for_update(node_id)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        for child_id in children_ids:
            ref_cache.add_entry(child_id, node_id,
                                ref_bug.tensor_cache.get_entry(child_id, node_id))
        # Test
        found_state, found_cache = bug.prepare_cache_for_update(node_id)
        self.assertTrue(found_cache.close_to(ref_cache))
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertEqual(ref_bug.state, found_state)
        self.assertEqual(id(found_state), id(bug.state))

    # TODO: When write tests for the other nodes, once all required methods are refactored

class TestPrepareForNodeUpdate(TestCase):
    """
    Tests that the caches and states are correctly prepared before a node is
    updated.
    """

    # Test for leafs
    def test_simple_c1(self):
        """
        Tests the method for the node c1 in the simple tree.
        """
        node_id = "c1"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_node = deepcopy(bug.state.nodes[node_id])
        ref_tensor = deepcopy(bug.ttns_dict[node_id].tensors[node_id])
        ref_state = deepcopy(bug.state)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        # Test
        found_state, found_cache = bug.prepare_for_node_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))
        found_node, found_tensor = found_state[node_id]
        self.assertEqual(id(found_state),id(bug.state))
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        # Other nodes haven't changed
        for key in bug.state.nodes:
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,found_state))

    def test_simple_c2(self):
        """
        Tests the method for the node c2 in the simple tree.
        """
        node_id = "c2"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_node = deepcopy(bug.state.nodes[node_id])
        ref_tensor = deepcopy(bug.ttns_dict[node_id].tensors[node_id])
        ref_state = deepcopy(bug.state)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        # Test
        found_state, found_cache = bug.prepare_for_node_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))
        found_node, found_tensor = found_state[node_id]
        self.assertEqual(id(found_state),id(bug.state))
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        # Other nodes haven't changed
        for key in bug.state.nodes:
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,found_state))

    def test_complicated_site2(self):
        """
        Tests the method for the node site2 in the complicated tree.
        """
        node_id = "site2"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_node = deepcopy(bug.state.nodes[node_id])
        ref_tensor = deepcopy(bug.ttns_dict[node_id].tensors[node_id])
        ref_state = deepcopy(bug.state)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        # Test
        found_state, found_cache = bug.prepare_for_node_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))
        found_node, found_tensor = found_state[node_id]
        self.assertEqual(id(found_state),id(bug.state))
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        # Other nodes haven't changed
        for key in bug.state.nodes:
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,found_state))

    def test_complicated_site7(self):
        """
        Tests the method for the node site7 in the complicated tree.
        """
        node_id = "site7"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site6")
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_node = deepcopy(bug.state.nodes[node_id])
        ref_tensor = deepcopy(bug.ttns_dict[node_id].tensors[node_id])
        ref_state = deepcopy(bug.state)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        # Test
        found_state, found_cache = bug.prepare_for_node_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))
        found_node, found_tensor = found_state[node_id]
        self.assertEqual(id(found_state),id(bug.state))
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        # Other nodes haven't changed
        for key in bug.state.nodes:
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,found_state))

    def test_complicated_site4(self):
        """
        Tests the method for the node site4 in the complicated tree.
        """
        node_id = "site4"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_node = deepcopy(bug.state.nodes[node_id])
        ref_tensor = deepcopy(bug.ttns_dict[node_id].tensors[node_id])
        ref_state = deepcopy(bug.state)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        # Test
        found_state, found_cache = bug.prepare_for_node_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))
        found_node, found_tensor = found_state[node_id]
        self.assertEqual(id(found_state),id(bug.state))
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        # Other nodes haven't changed
        for key in bug.state.nodes:
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,found_state))

    def test_complicated_site5(self):
        """
        Tests the method for the node site5 in the complicated tree.
        """
        node_id = "site5"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_node = deepcopy(bug.state.nodes[node_id])
        ref_tensor = deepcopy(bug.ttns_dict[node_id].tensors[node_id])
        ref_state = deepcopy(bug.state)
        ref_cache = deepcopy(bug.cache_dict[node_id])
        # Test
        found_state, found_cache = bug.prepare_for_node_update(node_id)
        self.assertIn(node_id,bug.cache_dict)
        self.assertEqual(id(found_cache), id(bug.cache_dict[node_id]))
        self.assertTrue(ref_cache.close_to(found_cache))
        found_node, found_tensor = found_state[node_id]
        self.assertEqual(id(found_state),id(bug.state))
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        # Other nodes haven't changed
        for key in bug.state.nodes:
            if key != node_id:
                self.assertTrue(ref_state.nodes_equal(key,found_state))

class TestGetOldTensorForNewBasisTensor(TestCase):
    """
    Tests if the correct tensor and node are pulled for the computation of 
    the new basis tensor.
    """

    # Test for leafs
    def test_simple_c1(self):
        """
        Tests the method for the node c1 in the simple tree.
        """
        node_id = "c1"
        bug = init_simple_bug()
        ref_state = deepcopy(bug.state)
        ref_node, ref_tensor = ref_state[node_id]
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        # Test
        found_node, found_tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        self.assertEqual(found_node.shape, found_tensor.shape)
        dim = found_node.shape[1]
        identity = eye(dim)
        self.assertTrue(allclose(identity,
                                 found_tensor.T.conj() @ found_tensor))


    def test_simple_c2(self):
        """
        Tests the method for the node c2 in the simple tree.
        """
        node_id = "c2"
        bug = init_simple_bug()
        ref_state = deepcopy(bug.state)
        ref_node, ref_tensor = ref_state[node_id]
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        # Test
        found_node, found_tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        self.assertEqual(found_node.shape, found_tensor.shape)
        dim = found_node.shape[1]
        identity = eye(dim)
        self.assertTrue(allclose(identity,
                                 found_tensor.T.conj() @ found_tensor))

    def test_complicated_site2(self):
        """
        Tests the method for the node site2 in the complicated tree.
        """
        node_id = "site2"
        bug = init_complicated_bug()
        ref_state = deepcopy(bug.state)
        ref_node, ref_tensor = ref_state[node_id]
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        # Test
        found_node, found_tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        self.assertEqual(found_node.shape, found_tensor.shape)
        dim = found_node.shape[1]
        identity = eye(dim)
        self.assertTrue(allclose(identity,
                                 found_tensor.T.conj() @ found_tensor))

    def test_complicated_site7(self):
        """
        Tests the method for the node site7 in the complicated tree.
        """
        node_id = "site7"
        bug = init_complicated_bug()
        ref_state = deepcopy(bug.state)
        ref_node, ref_tensor = ref_state[node_id]
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site6")
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        # Test
        found_node, found_tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        self.assertEqual(found_node.shape, found_tensor.shape)
        dim = found_node.shape[1]
        identity = eye(dim)
        self.assertTrue(allclose(identity,
                                 found_tensor.T.conj() @ found_tensor))

    def test_complicated_site4(self):
        """
        Tests the method for the node site4 in the complicated tree.
        """
        node_id = "site4"
        bug = init_complicated_bug()
        ref_state = deepcopy(bug.state)
        ref_node, ref_tensor = ref_state[node_id]
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        # Test
        found_node, found_tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        self.assertEqual(found_node.shape, found_tensor.shape)
        dim = found_node.shape[1]
        identity = eye(dim)
        self.assertTrue(allclose(identity,
                                 found_tensor.T.conj() @ found_tensor))

    def test_complicated_site5(self):
        """
        Tests the method for the node site5 in the complicated tree.
        """
        node_id = "site5"
        bug = init_complicated_bug()
        ref_state = deepcopy(bug.state)
        ref_node, ref_tensor = ref_state[node_id]
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        # Test
        found_node, found_tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))
        self.assertEqual(found_node.shape, found_tensor.shape)
        dim = found_node.shape[1]
        identity = eye(dim)
        self.assertTrue(allclose(identity,
                                 found_tensor.T.conj() @ found_tensor))

    # Test for non-leaf nodes abstractly
    ## We want to get the state tensor and no other one, so there is no need to
    ## Run all the preparation beforehand and this method should still work.
    def test_simple_root(self):
        """
        Tests the method for the root node in the simple tree.
        """
        node_id = "root"
        self.bug = init_simple_bug()
        ref_node, ref_tensor = deepcopy(self.bug.state[node_id])
        # Test
        found_node, found_tensor = self.bug._get_old_tensor_for_new_basis_tensor(node_id)
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))

    def complex_test(self, node_id: str):
        """
        Tests the method for the node node_id in the complicated tree.
        """
        self.bug = init_complicated_bug()
        ref_node, ref_tensor = deepcopy(self.bug.state[node_id])
        # Test
        found_node, found_tensor = self.bug._get_old_tensor_for_new_basis_tensor(node_id)
        self.assertEqual(ref_node, found_node)
        self.assertTrue(allclose(ref_tensor, found_tensor))

    def test_site0_complicated(self):
        """
        Tests the method for the node site0 in the complicated tree.
        """
        self.complex_test("site0")

    def test_site1_complicated(self):
        """
        Tests the method for the node site1 in the complicated tree.
        """
        self.complex_test("site1")

    def test_site3_complicated(self):
        """
        Tests the method for the node site3 in the complicated tree.
        """
        self.complex_test("site3")

    def test_site6_complicated(self):
        """
        Tests the method for the node site6 in the complicated tree.
        """
        self.complex_test("site6")

class TestComputeNewBasisTensor(TestCase):
    """
    Test if the new basis tensor is computed correctly.
    """

    # Test for leafs with random updated tensor
    def test_simple_c1(self):
        """
        Tests the method for the node c1 in the simple tree.
        """
        node_id = "c1"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        _, tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        rand_updated_tensor = crandn_like(tensor) # shape (3,3)
        # Test
        new_tensor = bug.compute_new_basis_tensor(node_id, rand_updated_tensor)
        self.assertEqual(tensor.shape, new_tensor.shape)
        self.assertTrue(allclose(new_tensor.T.conj() @ new_tensor,
                                 eye(tensor.shape[1])))

    def test_simple_c2(self):
        """
        Tests the method for the node c2 in the simple tree.
        """
        node_id = "c2"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        _, tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        rand_updated_tensor = crandn_like(tensor)
        # Test
        new_tensor = bug.compute_new_basis_tensor(node_id, rand_updated_tensor)
        self.assertEqual(tensor.shape, new_tensor.shape)
        self.assertTrue(allclose(new_tensor.T.conj() @ new_tensor,
                                 eye(tensor.shape[1])))

    def test_complicated_site2(self):
        """
        Tests the method for the node site2 in the complicated tree.
        """
        node_id = "site2"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        _, tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        rand_updated_tensor = crandn_like(tensor)
        # Test
        new_tensor = bug.compute_new_basis_tensor(node_id, rand_updated_tensor)
        self.assertEqual(2*tensor.shape[0], new_tensor.shape[0])
        self.assertEqual(tensor.shape[1], new_tensor.shape[1])
        self.assertTrue(allclose(new_tensor.T.conj() @ new_tensor,
                                 eye(tensor.shape[1])))

    def test_complicated_site7(self):
        """
        Tests the method for the node site7 in the complicated tree.
        """
        node_id = "site7"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site6")
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        _, tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        rand_updated_tensor = crandn_like(tensor)
        # Test
        new_tensor = bug.compute_new_basis_tensor(node_id, rand_updated_tensor)
        self.assertEqual(2*tensor.shape[0], new_tensor.shape[0])
        self.assertEqual(tensor.shape[1], new_tensor.shape[1])
        self.assertTrue(allclose(new_tensor.T.conj() @ new_tensor,
                                 eye(tensor.shape[1])))

    def test_complicated_site4(self):
        """
        Tests the method for the node site4 in the complicated tree.
        """
        node_id = "site4"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        _, tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        rand_updated_tensor = crandn_like(tensor)
        # Test
        new_tensor = bug.compute_new_basis_tensor(node_id, rand_updated_tensor)
        self.assertEqual(2*tensor.shape[0], new_tensor.shape[0])
        self.assertEqual(tensor.shape[1], new_tensor.shape[1])
        self.assertTrue(allclose(new_tensor.T.conj() @ new_tensor,
                                 eye(tensor.shape[1])))

    def test_complicated_site5(self):
        """
        Tests the method for the node site5 in the complicated tree.
        """
        node_id = "site5"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        _, tensor = bug._get_old_tensor_for_new_basis_tensor(node_id)
        rand_updated_tensor = crandn_like(tensor)
        # Test
        new_tensor = bug.compute_new_basis_tensor(node_id, rand_updated_tensor)
        self.assertEqual(2*tensor.shape[0], new_tensor.shape[0])
        self.assertEqual(tensor.shape[1], new_tensor.shape[1])
        self.assertTrue(allclose(new_tensor.T.conj() @ new_tensor,
                                 eye(tensor.shape[1])))

    # Test for non-leaf and non-root nodes
    def test_complicated_site6(self):
        """
        Tests the method for the node site6 in the complicated tree.
        """
        node_id = "site6"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        bug.prepare_for_node_update(node_id)
        _, ref_tensor = deepcopy(bug._get_old_tensor_for_new_basis_tensor(node_id))
        rand_updated_tensor = crandn_like(ref_tensor)
        # Test
        new_tensor = bug.compute_new_basis_tensor(node_id, rand_updated_tensor)
        self.assertEqual(2*ref_tensor.shape[0], new_tensor.shape[0])
        self.assertEqual(ref_tensor.shape[1:], new_tensor.shape[1:])
        dim = 2
        identity = eye(dim)
        found_ident = tensordot(new_tensor.conj(), new_tensor,
                                axes = ([1,2],[1,2]))
        self.assertTrue(allclose(identity, found_ident))

    def test_complicated_site3(self):
        """
        Tests the method for the node site3 in the complicated tree.
        """
        node_id = "site3"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        bug.prepare_for_node_update(node_id)
        _, ref_tensor = deepcopy(bug._get_old_tensor_for_new_basis_tensor(node_id))
        rand_updated_tensor = crandn_like(ref_tensor)
        # Test
        new_tensor = bug.compute_new_basis_tensor(node_id, rand_updated_tensor)
        self.assertEqual(2*ref_tensor.shape[0], new_tensor.shape[0])
        self.assertEqual(ref_tensor.shape[1:], new_tensor.shape[1:])
        dim = 2
        identity = eye(dim)
        found_ident = tensordot(new_tensor.conj(), new_tensor,
                                axes = ([1,2,3],[1,2,3]))
        self.assertTrue(allclose(identity, found_ident))

    # TODO: Write tests for the other nodes, once all required methods are adapted

class TestGetNodeAndTensorForComputingBasisChangeTensor(TestCase):
    """
    Tests if the correct node and tensor are used for the computation of the
    basis change tensor.
    """

    def test_simple_c1(self):
        """
        Tests the method for node c1 in the simple tree.
        """
        node_id = "c1"
        parent_id = "root"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        # Reference
        ref_bug = deepcopy(bug)
        correct_node = ref_bug.state.nodes[node_id]
        correct_tensor = ref_bug.ttns_dict[parent_id].tensors[node_id]
        # Test
        found_node, found_tensor = bug._get_node_and_tensor_for_basis_change_comp(node_id)
        self.assertEqual(correct_node,found_node)
        self.assertTrue(allclose(correct_tensor,found_tensor))

    def test_simple_c2(self):
        """
        Tests the method for node c2 in the simple tree.
        """
        node_id = "c2"
        parent_id = "root"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        # Reference
        ref_bug = deepcopy(bug)
        correct_node = ref_bug.state.nodes[node_id]
        correct_tensor = ref_bug.ttns_dict[parent_id].tensors[node_id]
        # Test
        found_node, found_tensor = bug._get_node_and_tensor_for_basis_change_comp(node_id)
        self.assertEqual(correct_node,found_node)
        self.assertTrue(allclose(correct_tensor,found_tensor))

class TestComputeBasisChangeTensor(TestCase):
    """
    Tests, if the basis change tensor is computed correctly.
    """

    def test_simple_c1(self):
        """
        Tests the method for node c1 in the simple tree.
        """
        node_id = "c1"
        parent_id = "root"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        random_new_basis_tensor = crandn(8,3)
        # Reference
        ref_bug = deepcopy(bug)
        ref_state_tensor = ref_bug.ttns_dict[parent_id].tensors[node_id]
        correct_tensor = (random_new_basis_tensor.conj() @ ref_state_tensor.T).T
        # Test
        found_tensor = bug.compute_basis_change_tensor(node_id,
                                                       random_new_basis_tensor)
        self.assertTrue(found_tensor.shape, (3,8))
        self.assertTrue(allclose(correct_tensor, found_tensor))

    def test_simple_c2(self):
        """
        Tests the method for node c2 in the simple tree.
        """
        node_id = "c2"
        parent_id = "root"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        random_new_basis_tensor = crandn(8,4)
        # Reference
        ref_bug = deepcopy(bug)
        ref_state_tensor = ref_bug.ttns_dict[parent_id].tensors[node_id]
        correct_tensor = (random_new_basis_tensor.conj() @ ref_state_tensor.T).T
        # Test
        found_tensor = bug.compute_basis_change_tensor(node_id,
                                                       random_new_basis_tensor)
        self.assertTrue(found_tensor.shape, (4,8))
        self.assertTrue(allclose(correct_tensor, found_tensor))

    def test_complicated_site2(self):
        """
        Tests the method for node site2 in the complicated tree.
        """
        node_id = "site2"
        parent_id = "site1"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        random_new_basis_tensor = crandn(3,2)
        # Prepare Reference
        ref_bug = deepcopy(bug)
        ref_state_tensor = ref_bug.ttns_dict[parent_id].tensors[node_id]
        correct_tensor = tensordot(ref_state_tensor,
                                   random_new_basis_tensor.conj(),
                                   axes=([1],[1]))
        # Test
        found_tensor = bug.compute_basis_change_tensor(node_id,
                                                       random_new_basis_tensor)
        self.assertTrue(found_tensor.shape, (1,3))
        self.assertTrue(allclose(correct_tensor, found_tensor))

    def test_complicated_site7(self):
        """
        Tests the method for node site7 in the complicated tree.
        """
        node_id = "site7"
        parent_id = "site6"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        random_new_basis_tensor = crandn(3,2)
        # Prepare Reference
        ref_bug = deepcopy(bug)
        ref_state_tensor = ref_bug.ttns_dict[parent_id].tensors[node_id]
        correct_tensor = tensordot(ref_state_tensor,
                                   random_new_basis_tensor.conj(),
                                   axes=([1],[1]))
        # Test
        found_tensor = bug.compute_basis_change_tensor(node_id,
                                                       random_new_basis_tensor)
        self.assertTrue(found_tensor.shape, (1,3))
        self.assertTrue(allclose(correct_tensor, found_tensor))


    def test_complicated_site4(self):
        """
        Tests the method for node site4 in the complicated tree.
        """
        node_id = "site4"
        parent_id = "site3"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        random_new_basis_tensor = crandn(3,2)
        # Prepare Reference
        ref_bug = deepcopy(bug)
        ref_state_tensor = ref_bug.ttns_dict[parent_id].tensors[node_id]
        correct_tensor = tensordot(ref_state_tensor,
                                   random_new_basis_tensor.conj(),
                                   axes=([1],[1]))
        # Test
        found_tensor = bug.compute_basis_change_tensor(node_id,
                                                       random_new_basis_tensor)
        self.assertTrue(found_tensor.shape, (1,3))
        self.assertTrue(allclose(correct_tensor, found_tensor))

    def test_complicated_site5(self):
        """
        Tests the method for node site5 in the complicated tree.
        """
        node_id = "site5"
        parent_id = "site3"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        bug.prepare_for_node_update(node_id)
        random_new_basis_tensor = crandn(3,2)
        # Prepare Reference
        ref_bug = deepcopy(bug)
        ref_state_tensor = ref_bug.ttns_dict[parent_id].tensors[node_id]
        correct_tensor = tensordot(ref_state_tensor,
                                   random_new_basis_tensor.conj(),
                                   axes=([1],[1]))
        # Test
        found_tensor = bug.compute_basis_change_tensor(node_id,
                                                       random_new_basis_tensor)
        self.assertTrue(found_tensor.shape, (1,3))
        self.assertTrue(allclose(correct_tensor, found_tensor))

    def test_complicated_site6(self):
        """
        Tests the method for node site6 in the complicated tree.
        """
        node_id = "site6"
        parent_id = "site0"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        bug.prepare_for_node_update(node_id)
        random_new_basis_tensor = crandn(3,2,2)
        # Prepare Reference
        ref_bug = deepcopy(bug)
        ref_state_tensor = ref_bug.ttns_dict[parent_id].tensors[node_id]
        child_id = "site7"
        cache_child = ref_bug.basis_change_cache.get_entry(child_id,
                                                           node_id)
        correct_tensor = tensordot(ref_state_tensor,
                                   cache_child,
                                   axes=([1],[0]))
        correct_tensor = tensordot(correct_tensor,
                                   random_new_basis_tensor.conj(),
                                   axes=([1,2],[2,1]))
        # Test
        found_tensor = bug.compute_basis_change_tensor(node_id,
                                                       random_new_basis_tensor)
        self.assertTrue(found_tensor.shape, (1,3))

    def test_complicated_site3(self):
        """
        Tests the method for node site3 in the complicated tree.
        """
        node_id = "site3"
        parent_id = "site1"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        bug.prepare_for_node_update(node_id)
        random_new_basis_tensor = crandn(3,2,2,2)
        # Prepare Reference
        ref_bug = deepcopy(bug)
        ref_state_tensor = ref_bug.ttns_dict[parent_id].tensors[node_id]
        correct_tensor = ref_state_tensor
        for child_id in ["site4","site5"]:
            cache_child = ref_bug.basis_change_cache.get_entry(child_id,
                                                                node_id)
            correct_tensor = tensordot(correct_tensor,
                                    cache_child,
                                    axes=([1],[0]))
        correct_tensor = tensordot(correct_tensor,
                                   random_new_basis_tensor.conj(),
                                   axes=([1,2,3],[3,1,2]))
        # Test
        found_tensor = bug.compute_basis_change_tensor(node_id,
                                                       random_new_basis_tensor)
        self.assertTrue(found_tensor.shape, (1,3))
        self.assertTrue(allclose(correct_tensor, found_tensor))

    # TODO: Write tests for the other nodes, once all required methods are adapted

class TestFindNewTensors(TestCase):
    """
    Tests, if the two new tensors to be added into the state are found correctly.0
    """

    # Test for leafs
    def test_simple_c1(self):
        """
        Tests the method for node c1 in the simple tree.
        """
        node_id = "c1"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_bug.prepare_for_node_update(node_id)
        updated_tensor = ref_bug.time_evolve_node(ref_bug.state[node_id],
                                                  ref_bug.hamiltonian[node_id],
                                                  ref_bug.cache_dict[node_id])
        correct_newb = ref_bug.compute_new_basis_tensor(node_id, updated_tensor)
        correct_bc = ref_bug.compute_basis_change_tensor(node_id, correct_newb)
        # Test
        found_newb, found_bc = bug.find_new_tensors(node_id)
        self.assertTrue(allclose(correct_newb, found_newb))
        self.assertTrue(allclose(correct_bc, found_bc))

    def test_simple_c2(self):
        """
        Tests the method for node c2 in the simple tree.
        """
        node_id = "c2"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_bug.prepare_for_node_update(node_id)
        updated_tensor = ref_bug.time_evolve_node(ref_bug.state[node_id],
                                                  ref_bug.hamiltonian[node_id],
                                                  ref_bug.cache_dict[node_id])
        correct_newb = ref_bug.compute_new_basis_tensor(node_id, updated_tensor)
        correct_bc = ref_bug.compute_basis_change_tensor(node_id, correct_newb)
        # Test
        found_newb, found_bc = bug.find_new_tensors(node_id)
        self.assertTrue(allclose(correct_newb, found_newb))
        self.assertTrue(allclose(correct_bc, found_bc))

    def test_complicated_site2(self):
        """
        Tests the method for node site2 in the complicated tree.
        """
        node_id = "site2"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_bug.prepare_for_node_update(node_id)
        updated_tensor = ref_bug.time_evolve_node(ref_bug.state[node_id],
                                                  ref_bug.hamiltonian[node_id],
                                                  ref_bug.cache_dict[node_id])
        correct_newb = ref_bug.compute_new_basis_tensor(node_id, updated_tensor)
        correct_bc = ref_bug.compute_basis_change_tensor(node_id, correct_newb)
        # Test
        found_newb, found_bc = bug.find_new_tensors(node_id)
        self.assertTrue(allclose(correct_newb, found_newb))
        self.assertTrue(allclose(correct_bc, found_bc))

    def test_complicated_site7(self):
        """
        Tests the method for node site7 in the complicated tree.
        """
        node_id = "site7"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site6")
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_bug.prepare_for_node_update(node_id)
        updated_tensor = ref_bug.time_evolve_node(ref_bug.state[node_id],
                                                  ref_bug.hamiltonian[node_id],
                                                  ref_bug.cache_dict[node_id])
        correct_newb = ref_bug.compute_new_basis_tensor(node_id, updated_tensor)
        correct_bc = ref_bug.compute_basis_change_tensor(node_id, correct_newb)
        # Test
        found_newb, found_bc = bug.find_new_tensors(node_id)
        self.assertTrue(allclose(correct_newb, found_newb))
        self.assertTrue(allclose(correct_bc, found_bc))

    def test_complicated_site4(self):
        """
        Tests the method for node site4 in the complicated tree.
        """
        node_id = "site4"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_bug.prepare_for_node_update(node_id)
        updated_tensor = ref_bug.time_evolve_node(ref_bug.state[node_id],
                                                  ref_bug.hamiltonian[node_id],
                                                  ref_bug.cache_dict[node_id])
        correct_newb = ref_bug.compute_new_basis_tensor(node_id, updated_tensor)
        correct_bc = ref_bug.compute_basis_change_tensor(node_id, correct_newb)
        # Test
        found_newb, found_bc = bug.find_new_tensors(node_id)
        self.assertTrue(allclose(correct_newb, found_newb))
        self.assertTrue(allclose(correct_bc, found_bc))

    def test_complicated_site5(self):
        """
        Tests the method for node site5 in the complicated tree.
        """
        node_id = "site5"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache("site3")
        bug.create_ttns_and_cache(node_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_bug.prepare_for_node_update(node_id)
        updated_tensor = ref_bug.time_evolve_node(ref_bug.state[node_id],
                                                  ref_bug.hamiltonian[node_id],
                                                  ref_bug.cache_dict[node_id])
        correct_newb = ref_bug.compute_new_basis_tensor(node_id, updated_tensor)
        correct_bc = ref_bug.compute_basis_change_tensor(node_id, correct_newb)
        # Test
        found_newb, found_bc = bug.find_new_tensors(node_id)
        self.assertTrue(allclose(correct_newb, found_newb))
        self.assertTrue(allclose(correct_bc, found_bc))

    # Non-leaf and non-root nodes
    def test_complicated_site6(self):
        """
        Tests the method for node site6 in the complicated tree.
        """
        node_id = "site6"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_bug.prepare_for_node_update(node_id)
        updated_tensor = ref_bug.time_evolve_node(ref_bug.state[node_id],
                                                  ref_bug.hamiltonian[node_id],
                                                  ref_bug.cache_dict[node_id])
        correct_newb = ref_bug.compute_new_basis_tensor(node_id, updated_tensor)
        correct_bc = ref_bug.compute_basis_change_tensor(node_id, correct_newb)
        # Test
        found_newb, found_bc = bug.find_new_tensors(node_id)
        self.assertTrue(allclose(correct_newb, found_newb))
        self.assertTrue(allclose(correct_bc, found_bc))

    def test_complicated_site3(self):
        """
        Tests the method for node site3 in the complicated tree.
        """
        node_id = "site3"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        ref_bug.prepare_for_node_update(node_id)
        updated_tensor = ref_bug.time_evolve_node(ref_bug.state[node_id],
                                                  ref_bug.hamiltonian[node_id],
                                                  ref_bug.cache_dict[node_id])
        correct_newb = ref_bug.compute_new_basis_tensor(node_id, updated_tensor)
        correct_bc = ref_bug.compute_basis_change_tensor(node_id, correct_newb)
        # Test
        found_newb, found_bc = bug.find_new_tensors(node_id)
        self.assertTrue(allclose(correct_newb, found_newb))
        self.assertTrue(allclose(correct_bc, found_bc))

    # TODO: Write tests for the other nodes, once all required methods are adapted

class TestReplaceNodeWithUpdatedBasis(TestCase):
    """
    Tests if a node is correctly replaced by two tensors.
    """

    def test_simple_c1(self):
        """
        Tests the method for node c1 in the simple tree.
        """
        node_id = "c1"
        parent_id = "root"
        bug = init_simple_bug()
        new_basis_tensor = crandn(5,3)
        basis_change_tensor = crandn(3,5)
        # Reference
        ref_nb = deepcopy(new_basis_tensor)
        ref_bc = deepcopy(basis_change_tensor)
        nb_node = Node(tensor=new_basis_tensor,
                       identifier=node_id)
        bc_node_id = basis_change_tensor_id(node_id)
        bc_node = Node(tensor=basis_change_tensor,
                          identifier=bc_node_id)
        bc_node.open_leg_to_parent(parent_id,0)
        bc_node.open_leg_to_child(node_id,1)
        nb_node.open_leg_to_parent(bc_node_id,0)
        # Test
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        self.assertIn(node_id, bug.state)
        self.assertTrue(allclose(ref_nb, bug.state[node_id][1]))
        self.assertEqual(nb_node, bug.state[node_id][0])
        self.assertIn(bc_node_id, bug.state)
        self.assertTrue(allclose(ref_bc, bug.state[bc_node_id][1]))
        self.assertEqual(bc_node, bug.state[bc_node_id][0])

    def test_simple_c2(self):
        """
        Tests the method for node c2 in the simple tree.
        """
        node_id = "c2"
        parent_id = "root"
        bug = init_simple_bug()
        new_basis_tensor = crandn(5,4)
        basis_change_tensor = crandn(4,5)
        # Reference
        ref_nb = deepcopy(new_basis_tensor)
        ref_bc = deepcopy(basis_change_tensor)
        nb_node = Node(tensor=new_basis_tensor,
                       identifier=node_id)
        bc_node_id = basis_change_tensor_id(node_id)
        bc_node = Node(tensor=basis_change_tensor,
                          identifier=bc_node_id)
        bc_node.open_leg_to_parent(parent_id,0)
        bc_node.open_leg_to_child(node_id,1)
        nb_node.open_leg_to_parent(bc_node_id,0)
        # Test
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        self.assertIn(node_id, bug.state)
        self.assertTrue(allclose(ref_nb, bug.state[node_id][1]))
        self.assertEqual(nb_node, bug.state[node_id][0])
        self.assertIn(bc_node_id, bug.state)
        self.assertTrue(allclose(ref_bc, bug.state[bc_node_id][1]))
        self.assertEqual(bc_node, bug.state[bc_node_id][0])

    def test_complicated_site2(self):
        """
        Tests the method for node site2 in the complicated tree.
        """
        node_id = "site2"
        parent_id = "site1"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        new_basis_tensor = crandn(3,2)
        basis_change_tensor = crandn(1,3)
        # Reference
        ref_nb = deepcopy(new_basis_tensor)
        ref_bc = deepcopy(basis_change_tensor)
        nb_node = Node(tensor=new_basis_tensor,
                       identifier=node_id)
        bc_node_id = basis_change_tensor_id(node_id)
        bc_node = Node(tensor=basis_change_tensor,
                          identifier=bc_node_id)
        bc_node.open_leg_to_parent(parent_id,0)
        bc_node.open_leg_to_child(node_id,1)
        nb_node.open_leg_to_parent(bc_node_id,0)
        # Test
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        self.assertIn(node_id, bug.state)
        self.assertTrue(allclose(ref_nb, bug.state[node_id][1]))
        self.assertEqual(nb_node, bug.state[node_id][0])
        self.assertIn(bc_node_id, bug.state)
        self.assertTrue(allclose(ref_bc, bug.state[bc_node_id][1]))
        self.assertEqual(bc_node, bug.state[bc_node_id][0])

    def test_complicated_site7(self):
        """
        Tests the method for node site7 in the complicated tree.
        """
        node_id = "site7"
        parent_id = "site6"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        new_basis_tensor = crandn(3,2)
        basis_change_tensor = crandn(1,3)
        # Reference
        ref_nb = deepcopy(new_basis_tensor)
        ref_bc = deepcopy(basis_change_tensor)
        nb_node = Node(tensor=new_basis_tensor,
                       identifier=node_id)
        bc_node_id = basis_change_tensor_id(node_id)
        bc_node = Node(tensor=basis_change_tensor,
                          identifier=bc_node_id)
        bc_node.open_leg_to_parent(parent_id,0)
        bc_node.open_leg_to_child(node_id,1)
        nb_node.open_leg_to_parent(bc_node_id,0)
        # Test
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        self.assertIn(node_id, bug.state)
        self.assertTrue(allclose(ref_nb, bug.state[node_id][1]))
        self.assertEqual(nb_node, bug.state[node_id][0])
        self.assertIn(bc_node_id, bug.state)
        self.assertTrue(allclose(ref_bc, bug.state[bc_node_id][1]))
        self.assertEqual(bc_node, bug.state[bc_node_id][0])

    def test_complicated_site4(self):
        """
        Tests the method for node site4 in the complicated tree.
        """
        node_id = "site4"
        parent_id = "site3"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        new_basis_tensor = crandn(3,2)
        basis_change_tensor = crandn(1,3)
        # Reference
        ref_nb = deepcopy(new_basis_tensor)
        ref_bc = deepcopy(basis_change_tensor)
        nb_node = Node(tensor=new_basis_tensor,
                       identifier=node_id)
        bc_node_id = basis_change_tensor_id(node_id)
        bc_node = Node(tensor=basis_change_tensor,
                          identifier=bc_node_id)
        bc_node.open_leg_to_parent(parent_id,0)
        bc_node.open_leg_to_child(node_id,1)
        nb_node.open_leg_to_parent(bc_node_id,0)
        # Test
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        self.assertIn(node_id, bug.state)
        self.assertTrue(allclose(ref_nb, bug.state[node_id][1]))
        self.assertEqual(nb_node, bug.state[node_id][0])
        self.assertIn(bc_node_id, bug.state)
        self.assertTrue(allclose(ref_bc, bug.state[bc_node_id][1]))
        self.assertEqual(bc_node, bug.state[bc_node_id][0])

    def test_complicated_site5(self):
        """
        Tests the method for node site5 in the complicated tree.
        """
        node_id = "site5"
        parent_id = "site3"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        new_basis_tensor = crandn(3,2)
        basis_change_tensor = crandn(1,3)
        # Reference
        ref_nb = deepcopy(new_basis_tensor)
        ref_bc = deepcopy(basis_change_tensor)
        nb_node = Node(tensor=new_basis_tensor,
                       identifier=node_id)
        bc_node_id = basis_change_tensor_id(node_id)
        bc_node = Node(tensor=basis_change_tensor,
                          identifier=bc_node_id)
        bc_node.open_leg_to_parent(parent_id,0)
        bc_node.open_leg_to_child(node_id,1)
        nb_node.open_leg_to_parent(bc_node_id,0)
        # Test
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        self.assertIn(node_id, bug.state)
        self.assertTrue(allclose(ref_nb, bug.state[node_id][1]))
        self.assertEqual(nb_node, bug.state[node_id][0])
        self.assertIn(bc_node_id, bug.state)
        self.assertTrue(allclose(ref_bc, bug.state[bc_node_id][1]))
        self.assertEqual(bc_node, bug.state[bc_node_id][0])

    # Non-leaf and non-root nodes
    def test_complicated_site6(self):
        """
        Tests the method for node site6 in the complicated tree.
        """
        node_id = "site6"
        parent_id = "site0"
        bug = init_complicated_bug()
        new_basis_tensor = crandn(5,2,2)
        basis_change_tensor = crandn(4,5)
        # Reference
        ref_nb = deepcopy(new_basis_tensor)
        ref_bc = deepcopy(basis_change_tensor)
        nb_node = Node(tensor=new_basis_tensor,
                       identifier=node_id)
        bc_node_id = basis_change_tensor_id(node_id)
        bc_node = Node(tensor=basis_change_tensor,
                          identifier=bc_node_id)
        bc_node.open_leg_to_parent(parent_id,0)
        bc_node.open_leg_to_child(node_id,1)
        nb_node.open_leg_to_parent(bc_node_id,0)
        child_id = "site7"
        nb_node.open_leg_to_child(child_id,1)
        # Test
        print(bug.state[node_id][1].shape)
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        self.assertIn(node_id, bug.state)
        self.assertTrue(allclose(ref_nb, bug.state[node_id][1]))
        self.assertEqual(nb_node, bug.state[node_id][0])
        self.assertIn(bc_node_id, bug.state)
        self.assertTrue(allclose(ref_bc, bug.state[bc_node_id][1]))
        self.assertEqual(bc_node, bug.state[bc_node_id][0])

    def test_complicated_site3(self):
        """
        Tests the method for node site3 in the complicated tree.
        """
        node_id = "site3"
        parent_id = "site1"
        bug = init_complicated_bug()
        new_basis_tensor = crandn(3,2,2,2)
        basis_change_tensor = crandn(5,3)
        # Reference
        ref_nb = deepcopy(new_basis_tensor)
        ref_bc = deepcopy(basis_change_tensor)
        nb_node = Node(tensor=new_basis_tensor,
                       identifier=node_id)
        bc_node_id = basis_change_tensor_id(node_id)
        bc_node = Node(tensor=basis_change_tensor,
                          identifier=bc_node_id)
        bc_node.open_leg_to_parent(parent_id,0)
        bc_node.open_leg_to_child(node_id,1)
        nb_node.open_leg_to_parent(bc_node_id,0)
        child_id = "site4"
        nb_node.open_leg_to_child(child_id,1)
        child_id = "site5"
        nb_node.open_leg_to_child(child_id,2)
        # Test
        print(bug.state[node_id][1].shape)
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        self.assertIn(node_id, bug.state)
        self.assertTrue(allclose(ref_nb, bug.state[node_id][1]))
        self.assertEqual(nb_node, bug.state[node_id][0])
        self.assertIn(bc_node_id, bug.state)
        self.assertTrue(allclose(ref_bc, bug.state[bc_node_id][1]))
        self.assertEqual(bc_node, bug.state[bc_node_id][0])

    # TODO: Write tests for the other nodes, once all required methods are adapted

class TestUpdateTensorCache(TestCase):
    """
    Tests if the cache of tensors is updated correctly.
    """

    def test_simple_c1(self):
        """
        Tests the method for node c1 in the simple tree.
        """
        node_id = "c1"
        parent_id = "root"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        # Reference
        ref_bug = deepcopy(bug)
        leaf_tensor = ref_bug.state[node_id][1]
        ham_tensor = ref_bug.hamiltonian[node_id][1]
        ref_tensor = tensordot(leaf_tensor,ham_tensor,axes=([1],[2]))
        ref_tensor = tensordot(ref_tensor,leaf_tensor.conj(),axes=([2],[1]))
        # Test
        bug.update_tensor_cache(node_id)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_tensor,
                                 bug.tensor_cache.get_entry(node_id,parent_id)))

    def test_simple_c2(self):
        """
        Tests the method for node c2 in the simple tree.
        """
        node_id = "c2"
        parent_id = "root"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        # Reference
        ref_bug = deepcopy(bug)
        leaf_tensor = ref_bug.state[node_id][1]
        ham_tensor = ref_bug.hamiltonian[node_id][1]
        ref_tensor = tensordot(leaf_tensor,ham_tensor,axes=([1],[2]))
        ref_tensor = tensordot(ref_tensor,leaf_tensor.conj(),axes=([2],[1]))
        # Test
        bug.update_tensor_cache(node_id)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_tensor,
                                 bug.tensor_cache.get_entry(node_id,parent_id)))

    def test_complicated_site2(self):
        """
        Tests the method for node site2 in the complicated tree.
        """
        node_id = "site2"
        parent_id = "site1"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        # Reference
        ref_bug = deepcopy(bug)
        leaf_tensor = ref_bug.state[node_id][1]
        ham_tensor = ref_bug.hamiltonian[node_id][1]
        ref_tensor = tensordot(leaf_tensor,ham_tensor,axes=([1],[2]))
        ref_tensor = tensordot(ref_tensor,leaf_tensor.conj(),axes=([2],[1]))
        # Test
        bug.update_tensor_cache(node_id)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_tensor,
                                 bug.tensor_cache.get_entry(node_id,parent_id)))

    def test_complicated_site7(self):
        """
        Tests the method for node site7 in the complicated tree.
        """
        node_id = "site7"
        parent_id = "site6"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        # Reference
        ref_bug = deepcopy(bug)
        leaf_tensor = ref_bug.state[node_id][1]
        ham_tensor = ref_bug.hamiltonian[node_id][1]
        ref_tensor = tensordot(leaf_tensor,ham_tensor,axes=([1],[2]))
        ref_tensor = tensordot(ref_tensor,leaf_tensor.conj(),axes=([2],[1]))
        # Test
        bug.update_tensor_cache(node_id)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_tensor,
                                 bug.tensor_cache.get_entry(node_id,parent_id)))

    def test_complicated_site4(self):
        """
        Tests the method for node site4 in the complicated tree.
        """
        node_id = "site4"
        parent_id = "site3"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        # Reference
        ref_bug = deepcopy(bug)
        leaf_tensor = ref_bug.state[node_id][1]
        ham_tensor = ref_bug.hamiltonian[node_id][1]
        ref_tensor = tensordot(leaf_tensor,ham_tensor,axes=([1],[2]))
        ref_tensor = tensordot(ref_tensor,leaf_tensor.conj(),axes=([2],[1]))
        # Test
        bug.update_tensor_cache(node_id)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_tensor,
                                 bug.tensor_cache.get_entry(node_id,parent_id)))

    def test_complicated_site5(self):
        """
        Tests the method for node site5 in the complicated tree.
        """
        node_id = "site5"
        parent_id = "site3"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        # Reference
        ref_bug = deepcopy(bug)
        leaf_tensor = ref_bug.state[node_id][1]
        ham_tensor = ref_bug.hamiltonian[node_id][1]
        ref_tensor = tensordot(leaf_tensor,ham_tensor,axes=([1],[2]))
        ref_tensor = tensordot(ref_tensor,leaf_tensor.conj(),axes=([2],[1]))
        # Test
        bug.update_tensor_cache(node_id)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_tensor,
                                 bug.tensor_cache.get_entry(node_id,parent_id)))

    # Non-leaf and non-root nodes
    def test_complicated_site6(self):
        """
        Tests the method for node site6 in the complicated tree.
        """
        node_id = "site6"
        parent_id = "site0"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        new_tensor, basis_change_tensor = bug.find_new_tensors(node_id)
        bug.replace_node_with_updated_basis(node_id,
                                            new_tensor,
                                            basis_change_tensor)
        # Reference
        ref_bug = deepcopy(bug)
        ham_tensor = ref_bug.hamiltonian[node_id][1]
        child_id = "site7"
        child_env = ref_bug.tensor_cache.get_entry(child_id,node_id)
        ref_tensor = tensordot(new_tensor,child_env,
                               axes=([1],[0]))
        ref_tensor = tensordot(ref_tensor,ham_tensor,
                               axes=([1,2],[3,1]))
        ref_tensor = tensordot(ref_tensor,new_tensor.conj(),
                               axes=([1,3],[1,2]))
        # Test
        bug.update_tensor_cache(node_id)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_tensor,
                                 bug.tensor_cache.get_entry(node_id,parent_id)))
        
    def test_complicated_site3(self):
        """
        Tests the method for node site3 in the complicated tree.
        """
        node_id = "site3"
        parent_id = "site1"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        bug.update_children(node_id)
        new_tensor, basis_change_tensor = bug.find_new_tensors(node_id)
        bug.replace_node_with_updated_basis(node_id,
                                            new_tensor,
                                            basis_change_tensor)
        # Reference
        ref_bug = deepcopy(bug)
        ham_tensor = ref_bug.hamiltonian[node_id][1]
        child_id = "site4"
        child_env = ref_bug.tensor_cache.get_entry(child_id,node_id)
        ref_tensor = tensordot(new_tensor,child_env,
                               axes=([1],[0]))
        child_id = "site5"
        child_env = ref_bug.tensor_cache.get_entry(child_id,node_id)
        ref_tensor = tensordot(ref_tensor,child_env,
                               axes=([1],[0]))
        ref_tensor = tensordot(ref_tensor,ham_tensor,
                               axes=([1,2,4],[4,1,2]))
        ref_tensor = tensordot(ref_tensor,new_tensor.conj(),
                               axes=([1,2,4],[1,2,3]))
        # Test
        bug.update_tensor_cache(node_id)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_tensor,
                                 bug.tensor_cache.get_entry(node_id,parent_id)))

    # TODO: Write tests for the other nodes, once all required methods are adapted

class TestUpdateBasisChangeCache(TestCase):
    """
    Tests if the cache of basis change tensors is updated correctly.
    """

    def test_simple_c1(self):
        """
        Tests the method for node c1 in the simple tree.
        """
        node_id = "c1"
        parent_id = "root"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        new_basis_tensor = crandn(5,3)
        basis_change_tensor = crandn(3,5)
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        # Reference
        ref_basis_change_tensor = deepcopy(basis_change_tensor)
        # Test
        bug.update_basis_change_cache(node_id, basis_change_tensor)
        self.assertTrue(bug.basis_change_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_basis_change_tensor,
                                 bug.basis_change_cache.get_entry(node_id,parent_id)))

    def test_simple_c2(self):
        """
        Tests the method for node c2 in the simple tree.
        """
        node_id = "c2"
        parent_id = "root"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        new_basis_tensor = crandn(5,4)
        basis_change_tensor = crandn(4,5)
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        # Reference
        ref_basis_change_tensor = deepcopy(basis_change_tensor)
        # Test
        bug.update_basis_change_cache(node_id, basis_change_tensor)
        self.assertTrue(bug.basis_change_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_basis_change_tensor,
                                 bug.basis_change_cache.get_entry(node_id,parent_id)))

    def test_complicated_site2(self):
        """
        Tests the method for node site2 in the complicated tree.
        """
        node_id = "site2"
        parent_id = "site1"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        new_basis_tensor = crandn(3,2)
        basis_change_tensor = crandn(1,3)
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        # Reference
        ref_basis_change_tensor = deepcopy(basis_change_tensor)
        # Test
        bug.update_basis_change_cache(node_id, basis_change_tensor)
        self.assertTrue(bug.basis_change_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_basis_change_tensor,
                                 bug.basis_change_cache.get_entry(node_id,parent_id)))

    def test_complicated_site7(self):
        """
        Tests the method for node site7 in the complicated tree.
        """
        node_id = "site7"
        parent_id = "site6"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        new_basis_tensor = crandn(3,2)
        basis_change_tensor = crandn(1,3)
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        # Reference
        ref_basis_change_tensor = deepcopy(basis_change_tensor)
        # Test
        bug.update_basis_change_cache(node_id, basis_change_tensor)
        self.assertTrue(bug.basis_change_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_basis_change_tensor,
                                 bug.basis_change_cache.get_entry(node_id,parent_id)))

    def test_complicated_site4(self):
        """
        Tests the method for node site4 in the complicated tree.
        """
        node_id = "site4"
        parent_id = "site3"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        new_basis_tensor = crandn(3,2)
        basis_change_tensor = crandn(1,3)
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        # Reference
        ref_basis_change_tensor = deepcopy(basis_change_tensor)
        # Test
        bug.update_basis_change_cache(node_id, basis_change_tensor)
        self.assertTrue(bug.basis_change_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_basis_change_tensor,
                                 bug.basis_change_cache.get_entry(node_id,parent_id)))

    def test_complicated_site5(self):
        """
        Tests the method for node site5 in the complicated tree.
        """
        node_id = "site5"
        parent_id = "site3"
        bug = init_complicated_bug(mode=RandomTTNSMode.TRIVIALVIRTUAL)
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(parent_id)
        bug.create_ttns_and_cache(node_id)
        new_basis_tensor = crandn(3,2)
        basis_change_tensor = crandn(1,3)
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        # Reference
        ref_basis_change_tensor = deepcopy(basis_change_tensor)
        # Test
        bug.update_basis_change_cache(node_id, basis_change_tensor)
        self.assertTrue(bug.basis_change_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_basis_change_tensor,
                                 bug.basis_change_cache.get_entry(node_id,parent_id)))

    # Test for non-leaf and non-root nodes
    def test_complicated_site6(self):
        """
        Tests the method for node site6 in the complicated tree.
        """
        node_id = "site6"
        parent_id = "site0"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(node_id)
        new_basis_tensor = crandn(5,2,2)
        basis_change_tensor = crandn(4,5)
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        # Reference
        ref_basis_change_tensor = deepcopy(basis_change_tensor)
        # Test
        bug.update_basis_change_cache(node_id, basis_change_tensor)
        self.assertTrue(bug.basis_change_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_basis_change_tensor,
                                 bug.basis_change_cache.get_entry(node_id,parent_id)))
        
    def test_complicated_site3(self):
        """
        Tests the method for node site3 in the complicated tree.
        """
        node_id = "site3"
        parent_id = "site1"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(node_id)
        new_basis_tensor = crandn(3,2,2,2)
        basis_change_tensor = crandn(5,3)
        bug.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        # Reference
        ref_basis_change_tensor = deepcopy(basis_change_tensor)
        # Test
        bug.update_basis_change_cache(node_id, basis_change_tensor)
        self.assertTrue(bug.basis_change_cache.contains(node_id,parent_id))
        self.assertTrue(allclose(ref_basis_change_tensor,
                                 bug.basis_change_cache.get_entry(node_id,parent_id)))


    # TODO: Write tests for the other nodes, once all required methods are adapted


class TestCleanUp(TestCase):
    """
    Tests if the clean up method works correctly.
    """

    def test_remove_one(self):
        """
        Test if it can remove the single entry from the dictionaries
        """
        node_id = "root"
        bug = init_simple_bug()
        bug.ttns_dict[node_id] = deepcopy(bug.state)
        bug.cache_dict[node_id] = deepcopy(SandwichCache(bug.ttns_dict[node_id],
                                                         bug.hamiltonian))
        bug.clean_up(node_id)
        self.assertEqual(0,len(bug.ttns_dict))
        self.assertEqual(0,len(bug.cache_dict))

    def test_remove_from_multiple(self):
        """
        Removes an entry when there are multiple entries.
        """
        node_id = "root"
        bug = init_simple_bug()
        bug.ttns_dict[node_id] = deepcopy(bug.state)
        bug.cache_dict[node_id] = SandwichCache(bug.ttns_dict[node_id],
                                                         bug.hamiltonian)
        node_id2 = "c1"
        ref_tree = deepcopy(bug.state)
        ref_cache = SandwichCache(bug.ttns_dict[node_id],
                                                bug.hamiltonian)
        bug.ttns_dict[node_id2] = ref_tree
        bug.cache_dict[node_id2] = ref_cache
        bug.clean_up(node_id)
        self.assertEqual(1,len(bug.ttns_dict))
        self.assertEqual(1,len(bug.cache_dict))
        self.assertIn(node_id2,bug.ttns_dict)
        self.assertIn(node_id2,bug.cache_dict)
        self.assertEqual(id(ref_tree),id(bug.ttns_dict[node_id2]))
        self.assertEqual(id(ref_cache),id(bug.cache_dict[node_id2]))

    def test_exceptions(self):
        """
        If the node isn't there it should raise an exception.
        """
        bug = init_simple_bug()
        self.assertRaises(KeyError,bug.clean_up,"c1")

class TestSubtreeUpdate(TestCase):
    """
    Tests if a whole subtree is correctly updated.

    This means the nodes of the subtree in the main state should be replaced
    by enlarged orthogonal tensors, their parents should be the basis change
    tensors, the main tensor cache should contain the new environment tensor
    and the basis change cache should contain the new basis change tensors.
    """

    # Test for leafs
    def test_simple_c1(self):
        """
        Tests the method for node c1 in the simple tree.
        """
        node_id = "c1"
        parent_id = "root"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        # Prepare reference
        ref_bug = deepcopy(bug)
        # Test
        bug.subtree_update(node_id)
        self.assertIn(node_id, bug.state)
        self.assertIn(basis_change_tensor_id(node_id), bug.state)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        # Check basis is orthogonal
        self.assertNotEqual(id(bug.state[node_id][1]),
                                  id(ref_bug.state[node_id][1]))
        nb_tensor = bug.state[node_id][1]
        self.assertTrue(allclose(eye(3),
                                 tensordot(nb_tensor,nb_tensor.conj(),axes=(1,1))))
        # Check that Tr[M^dagger M] = dim
        bc_tensor = bug.state[basis_change_tensor_id(node_id)][1]
        dim = 3
        self.assertTrue(allclose(dim,
                                 tensordot(bc_tensor,bc_tensor.conj(),axes=([0,1],[1,0]))))

    def test_simple_c2(self):
        """
        Tests the method for node c2 in the simple tree.
        """
        node_id = "c2"
        parent_id = "root"
        bug = init_simple_bug()
        bug.prepare_root_for_child_update()
        # Prepare reference
        ref_bug = deepcopy(bug)
        # Test
        bug.subtree_update(node_id)
        self.assertIn(node_id, bug.state)
        self.assertIn(basis_change_tensor_id(node_id), bug.state)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        # Check basis is orthogonal
        self.assertNotEqual(id(bug.state[node_id][1]),
                                  id(ref_bug.state[node_id][1]))
        nb_tensor = bug.state[node_id][1]
        self.assertTrue(allclose(eye(4),
                                 tensordot(nb_tensor,nb_tensor.conj(),axes=(1,1))))
        # Check that Tr[M^dagger M] = dim
        bc_tensor = bug.state[basis_change_tensor_id(node_id)][1]
        dim = 4
        self.assertTrue(allclose(dim,
                                 tensordot(bc_tensor,bc_tensor.conj(),axes=([0,1],[1,0]))))

    def test_complicated_site2(self):
        """
        Tests the method for node site2 in the complicated tree.
        """
        node_id = "site2"
        parent_id = "site1"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(parent_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        # Test
        bug.subtree_update(node_id)
        self.assertIn(node_id, bug.state)
        self.assertIn(basis_change_tensor_id(node_id), bug.state)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        # Check basis is orthogonal
        self.assertNotEqual(id(bug.state[node_id][1]),
                                  id(ref_bug.state[node_id][1]))
        nb_tensor = bug.state[node_id][1]
        self.assertTrue(allclose(eye(2),
                                 tensordot(nb_tensor,nb_tensor.conj(),axes=(1,1))))
        # Check that Tr[M^dagger M] = dim
        bc_tensor = bug.state[basis_change_tensor_id(node_id)][1]
        dim = 2
        self.assertTrue(allclose(dim,
                                 tensordot(bc_tensor,bc_tensor.conj(),axes=([0,1],[1,0]))))

    def test_complicated_site7(self):
        """
        Tests the method for node site7 in the complicated tree.
        """
        node_id = "site7"
        parent_id = "site6"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache(parent_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        # Test
        bug.subtree_update(node_id)
        self.assertIn(node_id, bug.state)
        self.assertIn(basis_change_tensor_id(node_id), bug.state)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        # Check basis is orthogonal
        self.assertNotEqual(id(bug.state[node_id][1]),
                                  id(ref_bug.state[node_id][1]))
        nb_tensor = bug.state[node_id][1]
        self.assertTrue(allclose(eye(2),
                                 tensordot(nb_tensor,nb_tensor.conj(),axes=(1,1))))
        # Check that Tr[M^dagger M] = dim
        bc_tensor = bug.state[basis_change_tensor_id(node_id)][1]
        dim = 2
        self.assertTrue(allclose(dim,
                                 tensordot(bc_tensor,bc_tensor.conj(),axes=([0,1],[1,0]))))

    def test_complicated_site4(self):
        """
        Tests the method for node site4 in the complicated tree.
        """
        node_id = "site4"
        parent_id = "site3"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(parent_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        # Test
        bug.subtree_update(node_id)
        self.assertIn(node_id, bug.state)
        self.assertIn(basis_change_tensor_id(node_id), bug.state)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        # Check basis is orthogonal
        self.assertNotEqual(id(bug.state[node_id][1]),
                                  id(ref_bug.state[node_id][1]))
        nb_tensor = bug.state[node_id][1]
        self.assertTrue(allclose(eye(2),
                                 tensordot(nb_tensor,nb_tensor.conj(),axes=(1,1))))
        # Check that Tr[M^dagger M] = dim
        bc_tensor = bug.state[basis_change_tensor_id(node_id)][1]
        dim = 2
        self.assertTrue(allclose(dim,
                                 tensordot(bc_tensor,bc_tensor.conj(),axes=([0,1],[1,0]))))

    def test_complicated_site5(self):
        """
        Tests the method for node site5 in the complicated tree.
        """
        node_id = "site5"
        parent_id = "site3"
        bug = init_complicated_bug()
        bug.prepare_root_for_child_update()
        bug.create_ttns_and_cache("site1")
        bug.create_ttns_and_cache(parent_id)
        # Prepare reference
        ref_bug = deepcopy(bug)
        # Test
        bug.subtree_update(node_id)
        self.assertIn(node_id, bug.state)
        self.assertIn(basis_change_tensor_id(node_id), bug.state)
        self.assertTrue(bug.tensor_cache.contains(node_id,parent_id))
        # Check basis is orthogonal
        self.assertNotEqual(id(bug.state[node_id][1]),
                                  id(ref_bug.state[node_id][1]))
        nb_tensor = bug.state[node_id][1]
        self.assertTrue(allclose(eye(2),
                                 tensordot(nb_tensor,nb_tensor.conj(),axes=(1,1))))
        # Check that Tr[M^dagger M] = dim
        bc_tensor = bug.state[basis_change_tensor_id(node_id)][1]
        dim = 2
        self.assertTrue(allclose(dim,
                                 tensordot(bc_tensor,bc_tensor.conj(),axes=([0,1],[1,0]))))



if __name__ == "__main__":
    unit_main()
