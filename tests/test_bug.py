import unittest

from copy import deepcopy

from numpy import tensordot, allclose

from pytreenet.time_evolution.bug import BUG
from pytreenet.random.random_ttns_and_ttno import small_ttns_and_ttno
from pytreenet.contractions.sandwich_caching import SandwichCache

class TestBugInitSimple(unittest.TestCase):

    def build_ref_cache(self, ref_ttns, ref_ttno) -> SandwichCache:
        """
        Build the correct cache to appear after initialisation.
        """
        ref_cache = SandwichCache(ref_ttns,ref_ttno)
        ref_cache.update_tree_cache("c1","root")
        ref_cache.update_tree_cache("c2","root")
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

class TestBUGSimple(unittest.TestCase):

    def setUp(self):
        self.ttns, self.ttno = small_ttns_and_ttno()
        self.bug = BUG(self.ttns, self.ttno, 0.1, 1.0, [])

    def test_build_effective_leaf_hamiltonian_c1(self):
        """
        Test the building of the effective leaf Hamiltonian for node c1.
        """
        # Move ort root -> c1
        self.bug.state.move_orthogonalization_center("c1")
        # Update cache
        self.bug.tensor_cache.update_tree_cache("root","c1")
        # Build effective leaf Hamiltonian
        ham_eff = self.bug.build_effective_leaf_hamiltonian("c1")
        # Build reference hamiltonian
        cache_tensor = self.bug.tensor_cache.get_entry("root","c1") # Shape (3,2,3)
        ham_tensor = self.bug.hamiltonian.tensors["c1"] # Shape (2,3,3)
        ref_ham_eff = tensordot(cache_tensor, ham_tensor,
                                axes=(1,0))
        ref_ham_eff = ref_ham_eff.transpose(1,2,0,3)
        ref_ham_eff = ref_ham_eff.reshape(9,9)
        self.assertTrue(allclose(ham_eff, ref_ham_eff))
    
    def test_build_effective_leaf_hamiltonian_c2(self):
        """
        Test the building of the effective leaf Hamiltonian for node c2.
        """
        # Move ort root -> c2
        self.bug.state.move_orthogonalization_center("c2")
        # Update cache
        self.bug.tensor_cache.update_tree_cache("root","c2")
        # Build effective leaf Hamiltonian
        ham_eff = self.bug.build_effective_leaf_hamiltonian("c2")
        # Build reference hamiltonian
        cache_tensor = self.bug.tensor_cache.get_entry("root","c2") # Shape (4,2,4)
        ham_tensor = self.bug.hamiltonian.tensors["c2"] # Shape (2,4,4)
        ref_ham_eff = tensordot(cache_tensor, ham_tensor,
                                axes=(1,0))
        ref_ham_eff = ref_ham_eff.transpose(1,2,0,3)
        ref_ham_eff = ref_ham_eff.reshape(16,16)
        self.assertTrue(allclose(ham_eff, ref_ham_eff))

if __name__ == "__main__":
    unittest.main()
