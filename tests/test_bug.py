import unittest

from copy import deepcopy, copy

from numpy import allclose
from scipy.linalg import expm

from pytreenet.time_evolution.bug import (BUG,
                                          basis_change_tensor_id,
                                          reverse_basis_change_tensor_id,
                                          compute_new_basis_tensor,
                                          compute_basis_change_tensor)
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.random.random_ttns import RandomTTNSMode

from pytreenet.random.random_matrices import crandn_like
from pytreenet.contractions.sandwich_caching import SandwichCache
from pytreenet.contractions.effective_hamiltonians import get_effective_single_site_hamiltonian

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
        self.time_step_size = 0.1
        self.bug = BUG(self.ttns, self.ttno,
                       self.time_step_size, 1.0,
                       [])

    def test_pull_tensor_from_old_state_c1(self):
        """
        Test the pull_tensor_from_old_state method with center c1.
        """
        node_id = "c1"
        # Set a different tensor for the old_state node
        old_tensor = self.bug.old_state.tensors[node_id]
        new_tensor = crandn_like(old_tensor)
        ref_tensor = deepcopy(new_tensor)
        self.bug.old_state.tensors[node_id] = new_tensor
        # Pull the tensor from the old state
        self.bug.pull_tensor_from_old_state(node_id)
        # Check the result
        found_tensor = self.bug.state.tensors[node_id]
        self.assertTrue(allclose(found_tensor, ref_tensor))

    def test_pull_tensor_from_old_state_c2(self):
        """
        Tests the pull_tensor_from_old_state method with center c2.
        """
        node_id = "c2"
        # Set a different tensor for the old_state node
        old_tensor = self.bug.old_state.tensors[node_id]
        new_tensor = crandn_like(old_tensor)
        ref_tensor = deepcopy(new_tensor)
        self.bug.old_state.tensors[node_id] = new_tensor
        # Pull the tensor from the old state
        self.bug.pull_tensor_from_old_state(node_id)
        # Check the result
        found_tensor = self.bug.state.tensors[node_id]
        self.assertTrue(allclose(found_tensor, ref_tensor))

    def test_pull_tensor_from_old_state_root(self):
        """
        Tests the pull_tensor_from_old_state method with center root.
        """
        node_id = "root"
        # Set a different tensor for the old_state node
        old_tensor = self.bug.old_state.tensors[node_id]
        new_tensor = crandn_like(old_tensor)
        ref_tensor = deepcopy(new_tensor)
        self.bug.old_state.tensors[node_id] = new_tensor
        # We need to modfiy the roots children
        for child_id in copy(self.bug.state.nodes[node_id].children):
            new_id = basis_change_tensor_id(child_id)
            self.bug.state.change_node_identifier(new_id, child_id)
        # Pull the tensor from the old state
        self.bug.pull_tensor_from_old_state(node_id)
        # Check the result
        found_tensor = self.bug.state.tensors[node_id]
        self.assertTrue(allclose(found_tensor, ref_tensor))

    def test_get_old_state_tensor_c1(self):
        """
        Tests to get the correct tensor from the old tensor with node c1.
        """
        node_id = "c1"
        ref_bug = deepcopy(self.bug)
        found = self.bug.get_old_state_tensor(node_id)
        # No changes in legs required
        correct = ref_bug.old_state.tensors[node_id]
        self.assertTrue(allclose(found, correct))


    def test_get_old_state_tensor_root(self):
        """
        Tests to get the correct tensor from the old tensor with node root.
        """
        node_id = "root"
        # We want to change the child order of the root node in the old state
        self.bug.old_state.move_orthogonalization_center("c1")
        self.bug.old_state.move_orthogonalization_center(node_id)
        # Old state (c2,c1), new state (c1,c2)
        ref_bug = deepcopy(self.bug)
        # Get the tensor
        found = self.bug.get_old_state_tensor(node_id)
        # The legs of the tensor need to be changed
        correct = ref_bug.old_state.tensors[node_id]
        correct = correct.transpose((1,0,2))
        self.assertTrue(allclose(found, correct))

    def test_time_evolve_node_c1(self):
        """
        Tests the time evolution of the c1 tensor.
        """
        node_id = "c1"
        # Prepare
        self.bug.old_state.move_orthogonalization_center(node_id)
        self.bug.update_tensor_cache_old_state("root", node_id)
        self.bug.pull_tensor_from_old_state(node_id)
        ref_bug = deepcopy(self.bug)
        # Time evolve the tensor
        found_tensor = self.bug.time_evolve_node(node_id)
        # Reference result
        heff = get_effective_single_site_hamiltonian(node_id,
                                                     ref_bug.state,
                                                     ref_bug.hamiltonian,
                                                     ref_bug.tensor_cache)
        expm_heff = expm(-1j * self.time_step_size * heff)
        state_tensor = ref_bug.state.tensors[node_id]
        vector = state_tensor.flatten()
        new_state_tensor_vector = expm_heff @ vector
        old_shape = state_tensor.shape
        new_state_tensor = new_state_tensor_vector.reshape(old_shape)
        # Check the result
        self.assertTrue(allclose(found_tensor, new_state_tensor))

    def test_time_evolve_node_c2(self):
        """
        Tests the time evolution of the c2 tensor.
        """
        node_id = "c2"
        # Prepare
        self.bug.old_state.move_orthogonalization_center(node_id)
        self.bug.update_tensor_cache_old_state("root", node_id)
        self.bug.pull_tensor_from_old_state(node_id)
        ref_bug = deepcopy(self.bug)
        # Time evolve the tensor
        found_tensor = self.bug.time_evolve_node(node_id)
        # Reference result
        heff = get_effective_single_site_hamiltonian(node_id,
                                                     ref_bug.state,
                                                     ref_bug.hamiltonian,
                                                     ref_bug.tensor_cache)
        expm_heff = expm(-1j * self.time_step_size * heff)
        state_tensor = ref_bug.state.tensors[node_id]
        vector = state_tensor.flatten()
        new_state_tensor_vector = expm_heff @ vector
        old_shape = state_tensor.shape
        new_state_tensor = new_state_tensor_vector.reshape(old_shape)
        # Check the result
        self.assertTrue(allclose(found_tensor, new_state_tensor))

    def test_time_evolve_node_root(self):
        """
        Tests the time evolution of the root tensor.
        """
        node_id = "root"
        # Prepare
        ref_bug = deepcopy(self.bug)
        # Time evolve the tensor
        found_tensor = self.bug.time_evolve_node(node_id)
        # Reference result
        heff = get_effective_single_site_hamiltonian(node_id,
                                                     ref_bug.state,
                                                     ref_bug.hamiltonian,
                                                     ref_bug.tensor_cache)
        expm_heff = expm(-1j * self.time_step_size * heff)
        state_tensor = ref_bug.state.tensors[node_id]
        vector = state_tensor.flatten()
        new_state_tensor_vector = expm_heff @ vector
        old_shape = state_tensor.shape
        new_state_tensor = new_state_tensor_vector.reshape(old_shape)
        # Check the result
        self.assertTrue(allclose(found_tensor, new_state_tensor))

    def test_update_non_root_c1(self):
        """
        Tests to update the node c1.
        """
        node_id = "c1"
        parent_id = "root"
        # Prepare
        self.bug.old_state.move_orthogonalization_center(node_id)
        self.bug.update_tensor_cache_old_state(parent_id, node_id)
        self.bug.pull_tensor_from_old_state(node_id)
        ref_bug = deepcopy(self.bug)
        # Update the node
        self.bug.update_non_root(node_id)
        # Reference result
        updated_tensor = ref_bug.time_evolve_node(node_id)
        ref_bug.old_state.move_orthogonalization_center(parent_id)
        ref_bug.update_tensor_cache_old_state(parent_id, node_id)
        old_tensor = ref_bug.old_state.tensors[node_id]
        node = ref_bug.state.nodes[node_id]
        new_basis_tensor = compute_new_basis_tensor(node,
                                                    old_tensor,
                                                    updated_tensor)	
        basis_change_tensor = compute_basis_change_tensor(old_tensor,
                                                          new_basis_tensor)
        ref_bug.replace_node_with_updated_basis(node_id, new_basis_tensor,
                                                basis_change_tensor)
        # Check the result
        self.assertTrue(allclose(self.bug.state.tensors[node_id],
                                 ref_bug.state.tensors[node_id]))
        self.assertTrue(allclose(self.bug.state.tensors[basis_change_tensor_id(node_id)],
                                 ref_bug.state.tensors[basis_change_tensor_id(node_id)]))

    def test_update_non_root_c2(self):
        """
        Tests to update the node c2.
        """
        node_id = "c2"
        parent_id = "root"
        # Prepare
        self.bug.old_state.move_orthogonalization_center(node_id)
        self.bug.update_tensor_cache_old_state(parent_id, node_id)
        self.bug.pull_tensor_from_old_state(node_id)
        ref_bug = deepcopy(self.bug)
        # Update the node
        self.bug.update_non_root(node_id)
        # Reference result
        updated_tensor = ref_bug.time_evolve_node(node_id)
        ref_bug.old_state.move_orthogonalization_center(parent_id)
        ref_bug.update_tensor_cache_old_state(parent_id, node_id)
        old_tensor = ref_bug.old_state.tensors[node_id]
        node = ref_bug.state.nodes[node_id]
        new_basis_tensor = compute_new_basis_tensor(node,
                                                    old_tensor,
                                                    updated_tensor)	
        basis_change_tensor = compute_basis_change_tensor(old_tensor,
                                                          new_basis_tensor)
        ref_bug.replace_node_with_updated_basis(node_id, new_basis_tensor,
                                                basis_change_tensor)
        # Check the result
        self.assertTrue(allclose(self.bug.state.tensors[node_id],
                                 ref_bug.state.tensors[node_id]))
        self.assertTrue(allclose(self.bug.state.tensors[basis_change_tensor_id(node_id)],
                                 ref_bug.state.tensors[basis_change_tensor_id(node_id)]))

class TestBUGComplicated(unittest.TestCase):

    def setUp(self):
        self.ttns, self.ttno = big_ttns_and_ttno(mode=RandomTTNSMode.DIFFVIRT)
        self.bug = BUG(self.ttns, self.ttno, 0.1, 1.0, [])

    def test_pull_tensor_site0(self):
        """
        Test the pull_tensor_from_old_state method with center site0.
        """
        node_id = "site0"
        # Set a different tensor for the old_state node
        old_tensor = self.bug.old_state.tensors[node_id]
        new_tensor = crandn_like(old_tensor)
        ref_tensor = deepcopy(new_tensor)
        self.bug.old_state.tensors[node_id] = new_tensor
        # We need to modfiy the node's children
        for child_id in copy(self.bug.state.nodes[node_id].children):
            new_id = basis_change_tensor_id(child_id)
            self.bug.state.change_node_identifier(new_id, child_id)
        # Pull the tensor from the old state
        self.bug.pull_tensor_from_old_state(node_id)
        # Check the result
        found_tensor = self.bug.state.tensors[node_id]
        self.assertTrue(allclose(found_tensor, ref_tensor))

    def test_pull_tensor_site1(self):
        """
        Test the pull_tensor_from_old_state method with center site1.
        """
        node_id = "site1"
        # Set a different tensor for the old_state node
        old_tensor = self.bug.old_state.tensors[node_id]
        new_tensor = crandn_like(old_tensor)
        ref_tensor = deepcopy(new_tensor)
        self.bug.old_state.tensors[node_id] = new_tensor
        # We need to modfiy the node's children
        for child_id in copy(self.bug.state.nodes[node_id].children):
            new_id = basis_change_tensor_id(child_id)
            self.bug.state.change_node_identifier(new_id, child_id)
        # Pull the tensor from the old state
        self.bug.pull_tensor_from_old_state(node_id)
        # Check the result
        found_tensor = self.bug.state.tensors[node_id]
        self.assertTrue(allclose(found_tensor, ref_tensor))

    def test_pull_tensor_site2(self):
        """
        Test the pull_tensor_from_old_state method with center site2.
        """
        node_id = "site2"
        # Set a different tensor for the old_state node
        old_tensor = self.bug.old_state.tensors[node_id]
        new_tensor = crandn_like(old_tensor)
        ref_tensor = deepcopy(new_tensor)
        self.bug.old_state.tensors[node_id] = new_tensor
        # Pull the tensor from the old state
        self.bug.pull_tensor_from_old_state(node_id)
        # Check the result
        found_tensor = self.bug.state.tensors[node_id]
        self.assertTrue(allclose(found_tensor, ref_tensor))

    def test_pull_tensor_site3(self):
        """
        Test the pull_tensor_from_old_state method with center site3.
        """
        node_id = "site3"
        # Set a different tensor for the old_state node
        old_tensor = self.bug.old_state.tensors[node_id]
        new_tensor = crandn_like(old_tensor)
        ref_tensor = deepcopy(new_tensor)
        self.bug.old_state.tensors[node_id] = new_tensor
        # We need to modfiy the node's children
        for child_id in copy(self.bug.state.nodes[node_id].children):
            new_id = basis_change_tensor_id(child_id)
            self.bug.state.change_node_identifier(new_id, child_id)
        # Pull the tensor from the old state
        self.bug.pull_tensor_from_old_state(node_id)
        # Check the result
        found_tensor = self.bug.state.tensors[node_id]
        self.assertTrue(allclose(found_tensor, ref_tensor))

    def test_pull_tensor_site4(self):
        """
        Test the pull_tensor_from_old_state method with center site4.
        """
        node_id = "site4"
        # Set a different tensor for the old_state node
        old_tensor = self.bug.old_state.tensors[node_id]
        new_tensor = crandn_like(old_tensor)
        ref_tensor = deepcopy(new_tensor)
        self.bug.old_state.tensors[node_id] = new_tensor
        # Pull the tensor from the old state
        self.bug.pull_tensor_from_old_state(node_id)
        # Check the result
        found_tensor = self.bug.state.tensors[node_id]
        self.assertTrue(allclose(found_tensor, ref_tensor))

    def test_pull_tensor_site5(self):
        """
        Test the pull_tensor_from_old_state method with center site5.
        """
        node_id = "site5"
        # Set a different tensor for the old_state node
        old_tensor = self.bug.old_state.tensors[node_id]
        new_tensor = crandn_like(old_tensor)
        ref_tensor = deepcopy(new_tensor)
        self.bug.old_state.tensors[node_id] = new_tensor
        # Pull the tensor from the old state
        self.bug.pull_tensor_from_old_state(node_id)
        # Check the result
        found_tensor = self.bug.state.tensors[node_id]
        self.assertTrue(allclose(found_tensor, ref_tensor))

    def test_pull_tensor_site6(self):
        """
        Test the pull_tensor_from_old_state method with center site6.
        """
        node_id = "site6"
        # Set a different tensor for the old_state node
        old_tensor = self.bug.old_state.tensors[node_id]
        new_tensor = crandn_like(old_tensor)
        ref_tensor = deepcopy(new_tensor)
        self.bug.old_state.tensors[node_id] = new_tensor
        # We need to modfiy the node's children
        for child_id in copy(self.bug.state.nodes[node_id].children):
            new_id = basis_change_tensor_id(child_id)
            self.bug.state.change_node_identifier(new_id, child_id)
        # Pull the tensor from the old state
        self.bug.pull_tensor_from_old_state(node_id)
        # Check the result
        found_tensor = self.bug.state.tensors[node_id]
        self.assertTrue(allclose(found_tensor, ref_tensor))

    def test_pull_tensor_site7(self):
        """
        Test the pull_tensor_from_old_state method with center site7.
        """
        node_id = "site7"
        # Set a different tensor for the old_state node
        old_tensor = self.bug.old_state.tensors[node_id]
        new_tensor = crandn_like(old_tensor)
        ref_tensor = deepcopy(new_tensor)
        self.bug.old_state.tensors[node_id] = new_tensor
        # Pull the tensor from the old state
        self.bug.pull_tensor_from_old_state(node_id)
        # Check the result
        found_tensor = self.bug.state.tensors[node_id]
        self.assertTrue(allclose(found_tensor, ref_tensor))

    def test_get_old_state_tensor_site0_no_perm(self):
        """
        Tests to get the correct tensor from the old tensor with node site0.
        """
        node_id = "site0"
        ref_bug = deepcopy(self.bug)
        found = self.bug.get_old_state_tensor(node_id)
        # No changes in legs required
        correct = ref_bug.old_state.tensors[node_id]
        self.assertTrue(allclose(found, correct))

    def test_get_old_state_tensor_site0_perm(self):
        """
        Tests to get the correct tensor from the old tensor with node site0.
        The children are permuted against each other.
        """
        node_id = "site0"
        # We want to change the leg order in the old state
        self.bug.old_state.move_orthogonalization_center("site1")
        self.bug.old_state.move_orthogonalization_center("site0")
        ref_bug = deepcopy(self.bug)
        found = self.bug.get_old_state_tensor(node_id)
        # The legs of the tensor need to be changed
        correct = ref_bug.old_state.tensors[node_id]
        correct = correct.transpose((1,0,2))
        self.assertTrue(allclose(found, correct))

    def test_get_old_state_tensor_site1_no_perm(self):
        """
        Tests to get the correct tensor from the old tensor with node site1.
        """
        node_id = "site1"
        ref_bug = deepcopy(self.bug)
        found = self.bug.get_old_state_tensor(node_id)
        # No changes in legs required
        correct = ref_bug.old_state.tensors[node_id]
        self.assertTrue(allclose(found, correct))

    def test_get_old_state_tensor_site1_perm(self):
        """
        Tests to get the correct tensor from the old tensor with node site1.
        The children are permuted against each other.
        """
        node_id = "site1"
        # We want to change the leg order in the old state
        self.bug.old_state.move_orthogonalization_center("site1")
        self.bug.old_state.move_orthogonalization_center("site2")
        self.bug.old_state.move_orthogonalization_center("site0")
        ref_bug = deepcopy(self.bug)
        found = self.bug.get_old_state_tensor(node_id)
        # The legs of the tensor need to be changed
        correct = ref_bug.old_state.tensors[node_id]
        correct = correct.transpose((0,2,1,3))
        self.assertTrue(allclose(found, correct))

    def test_get_old_state_tensor_site3_no_perm(self):
        """
        Tests to get the correct tensor from the old tensor with node site3.
        """
        node_id = "site3"
        ref_bug = deepcopy(self.bug)
        found = self.bug.get_old_state_tensor(node_id)
        # No changes in legs required
        correct = ref_bug.old_state.tensors[node_id]
        self.assertTrue(allclose(found, correct))

    def test_get_old_state_tensor_site3_perm(self):
        """
        Tests to get the correct tensor from the old tensor with node site3.
        The children are permuted against each other.
        """
        node_id = "site3"
        # We want to change the leg order in the old state
        self.bug.old_state.move_orthogonalization_center("site4")
        self.bug.old_state.move_orthogonalization_center("site3")
        ref_bug = deepcopy(self.bug)
        found = self.bug.get_old_state_tensor(node_id)
        # The legs of the tensor need to be changed
        correct = ref_bug.old_state.tensors[node_id]
        correct = correct.transpose((0,2,1,3))
        self.assertTrue(allclose(found, correct))

    def test_get_old_state_tensor_site6(self):
        """
        Tests to get the correct tensor from the old tensor with node site6.
        """
        node_id = "site6"
        ref_bug = deepcopy(self.bug)
        found = self.bug.get_old_state_tensor(node_id)
        # No changes in legs required
        correct = ref_bug.old_state.tensors[node_id]
        self.assertTrue(allclose(found, correct))

    def test_time_evolve_site0(self):
        """
        Test the time evolution of the site0 tensor.
        """
        node_id = "site0"
        # Prepare
        ref_bug = deepcopy(self.bug)
        # Time evolve the tensor
        found_tensor = self.bug.time_evolve_node(node_id)
        # Reference result
        heff = get_effective_single_site_hamiltonian(node_id,
                                                     ref_bug.state,
                                                     ref_bug.hamiltonian,
                                                     ref_bug.tensor_cache)
        expm_heff = expm(-1j * self.bug.time_step_size * heff)
        state_tensor = ref_bug.state.tensors[node_id]
        vector = state_tensor.flatten()
        new_state_tensor_vector = expm_heff @ vector
        old_shape = state_tensor.shape
        new_state_tensor = new_state_tensor_vector.reshape(old_shape)
        # Check the result
        self.assertTrue(allclose(found_tensor, new_state_tensor))

    def test_time_evolve_site1(self):
        """
        Test the time evolution of the site1 tensor.

        A lot would happen before the tensor is actually time-evolved, so
        we need to accunt for that.
        """
        node_id = "site1"
        # Prepare
        self.bug.old_state.move_orthogonalization_center(node_id)
        self.bug.update_tensor_cache_old_state("site0", node_id)
        for child_id in copy(self.bug.state.nodes[node_id].children):
            # In the algorithm the children are changed to the basis
            # of the new state. We need to do the same here.
            new_id = basis_change_tensor_id(child_id)
            self.bug.state.change_node_identifier(new_id, child_id)
        self.bug.pull_tensor_from_old_state(node_id)
        for child_id in copy(self.bug.state.nodes[node_id].children):
            # In the algorithm the basis change tensors would be contracted,
            # so the old children identifiers would reapper
            new_id = reverse_basis_change_tensor_id(child_id)
            self.bug.state.change_node_identifier(new_id, child_id)
        ref_bug = deepcopy(self.bug)
        # Time evolve the tensor
        found_tensor = self.bug.time_evolve_node(node_id)
        # Reference result
        heff = get_effective_single_site_hamiltonian(node_id,
                                                     ref_bug.state,
                                                     ref_bug.hamiltonian,
                                                     ref_bug.tensor_cache)
        expm_heff = expm(-1j * self.bug.time_step_size * heff)
        state_tensor = ref_bug.state.tensors[node_id]
        vector = state_tensor.flatten()
        new_state_tensor_vector = expm_heff @ vector
        old_shape = state_tensor.shape
        new_state_tensor = new_state_tensor_vector.reshape(old_shape)
        # Check the result
        self.assertTrue(allclose(found_tensor, new_state_tensor))


if __name__ == "__main__":
    unittest.main()
