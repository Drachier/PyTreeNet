import unittest

from copy import deepcopy, copy

from numpy import allclose
from scipy.linalg import expm

from pytreenet.time_evolution.bug import BUG
from pytreenet.time_evolution.time_evo_util.bug_util import (basis_change_tensor_id,
                                                            reverse_basis_change_tensor_id)
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.random.random_ttns import RandomTTNSMode

from pytreenet.contractions.sandwich_caching import SandwichCache
from pytreenet.contractions.effective_hamiltonians import get_effective_single_site_hamiltonian
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict

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

class TestBUGSimple(unittest.TestCase):

    def setUp(self):
        self.ttns, self.ttno = small_ttns_and_ttno()
        self.time_step_size = 0.1
        self.bug = BUG(self.ttns, self.ttno,
                       self.time_step_size, 1.0,
                       [])

    def test_tree_update(self):
        self.bug.tree_update()

class TestBUGComplicated(unittest.TestCase):

    def setUp(self):
        self.ttns, self.ttno = big_ttns_and_ttno(mode=RandomTTNSMode.DIFFVIRT)
        self.bug = BUG(self.ttns, self.ttno, 0.1, 1.0, [])

    def test_tree_update(self):
        self.bug.tree_update()
        self.assertEqual(self.bug.ttns_dict, {})
        self.assertEqual(self.bug.cache_dict, {})
        self.assertEqual(self.bug.basis_change_cache, PartialTreeCachDict())

    def test_truncation(self):
        self.bug.truncation()

    def test_one_time_step(self):
        print("-"*80)
        self.bug.run_one_time_step()

if __name__ == "__main__":
    unittest.main()
