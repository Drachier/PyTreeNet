"""
This module contains tests for the effective time evolution functionality.
"""

import pytest
import unittest

import numpy as np
from scipy.linalg import expm

from pytreenet.random.random_node import random_tensor_node
from pytreenet.random.random_matrices import crandn
from pytreenet.time_evolution.time_evolution import TimeEvoMode, EvoDirection
from pytreenet.contractions.effective_hamiltonians import (get_effective_bond_hamiltonian_nodes,
                                                           get_effective_single_site_hamiltonian_nodes)
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.time_evolution.time_evo_util.effective_time_evolution import (effective_bond_evolution,
                                                                             effective_single_site_evolution,
                                                                             effective_two_site_evolution)

## Tests for effective bond evolution
@pytest.mark.parametrize("mode", TimeEvoMode.scipy_modes())
def test_action_based_bond_evolution(mode):
    """
    Test the bond evolution using action-based method.
    """
    node_id = "state_node"
    # Define state node
    state_node, state_tensor = random_tensor_node((4,5),
                                                  identifier=node_id)
    parent_id = "parent_node"
    child_id = "child_node"
    state_node.open_leg_to_parent(parent_id, 0)
    state_node.open_leg_to_child(child_id, 1)
    # Define cache
    cache = PartialTreeCachDict()
    parent_tensor = crandn(4,6,4)
    child_tensor = crandn(5,6,5)
    cache.add_entry(parent_id, child_id, parent_tensor)
    cache.add_entry(child_id, parent_id, child_tensor)
    # Parameters
    time_step_size = 0.1
    rtol = 1e-10
    atol = 1e-10
    # Reference evolution
    ref_ham = get_effective_bond_hamiltonian_nodes(state_node, cache)
    ref_updated = expm(-1j * time_step_size * ref_ham) @ state_tensor.flatten()
    ref_updated = ref_updated.reshape(4,5)
    # Perform evolution
    found_updated = effective_bond_evolution(state_tensor,
                                             state_node,
                                             time_step_size,
                                             cache,
                                             mode=mode,
                                             rtol=rtol,
                                             atol=atol)
    # Check if the evolution is correct
    np.testing.assert_allclose(found_updated, ref_updated)

@pytest.mark.parametrize("mode", [mode for mode in TimeEvoMode if not mode.is_scipy()])
def test_matrix_based_bond_evolution(mode):
    """
    Test the bond evolution using action-based method.
    """
    node_id = "state_node"
    # Define state node
    state_node, state_tensor = random_tensor_node((4,5),
                                                  identifier=node_id)
    parent_id = "parent_node"
    child_id = "child_node"
    state_node.open_leg_to_parent(parent_id, 0)
    state_node.open_leg_to_child(child_id, 1)
    # Define cache
    cache = PartialTreeCachDict()
    parent_tensor = crandn(4,6,4)
    child_tensor = crandn(5,6,5)
    cache.add_entry(parent_id, child_id, parent_tensor)
    cache.add_entry(child_id, parent_id, child_tensor)
    # Parameters
    time_step_size = 0.1
    # Reference evolution
    ref_ham = get_effective_bond_hamiltonian_nodes(state_node, cache)
    ref_updated = expm(-1j * time_step_size * ref_ham) @ state_tensor.flatten()
    ref_updated = ref_updated.reshape(4,5)
    # Perform evolution
    found_updated = effective_bond_evolution(state_tensor,
                                             state_node,
                                             time_step_size,
                                             cache,
                                             mode=mode)
    # Check if the evolution is correct
    np.testing.assert_allclose(found_updated, ref_updated)

## Tests for effective single site evolution
@pytest.mark.parametrize("mode", TimeEvoMode.scipy_modes())
def test_action_based_single_site_evolution(mode):
    """
    Test the single site evolution using action-based method.

    Note that the solvers seem very unstable for a randomly generated problem,
    so we are very lenient with the tolerances.
    """
    node_id = "main_node"
    # Define state node
    state_node, state_tensor = random_tensor_node((4,6,5,2),
                                                    identifier=node_id)
    parent_id = "parent_node"
    child1_id = "child1_node"
    child2_id = "child2_node"
    state_node.open_leg_to_parent(parent_id, 0)
    state_node.open_leg_to_child(child1_id, 1)
    state_node.open_leg_to_child(child2_id, 2)
    # Define Hamiltonian node
    ham_node, ham_tensor = random_tensor_node((7,8,9,2,2),
                                                identifier=node_id)
    ham_node.open_leg_to_parent(parent_id, 0)
    ham_node.open_leg_to_child(child2_id, 1)
    ham_node.open_leg_to_child(child1_id, 2)
    # Define cache
    cache = PartialTreeCachDict()
    cache.add_entry(parent_id, node_id, crandn(4,7,4))
    cache.add_entry(child1_id, node_id, crandn(6,9,6))
    cache.add_entry(child2_id, node_id, crandn(5,8,5))
    # Parameters
    time_step_size = 0.001
    rtol = 1e-5
    atol = 1e-5
    # Reference evolution
    ref_ham = get_effective_single_site_hamiltonian_nodes(state_node,
                                                          ham_node,
                                                          ham_tensor,
                                                          cache)
    ref_updated = expm(-1j * time_step_size * ref_ham) @ state_tensor.flatten()
    ref_updated = ref_updated.reshape(4,6,5,2)
    # Perform evolution
    found_updated = effective_single_site_evolution(state_tensor,
                                                    state_node,
                                                    ham_tensor,
                                                    ham_node,
                                                    time_step_size,
                                                    cache,
                                                    mode=mode,
                                                    rtol=rtol,
                                                    atol=atol)
    # Check if the evolution is correct
    np.testing.assert_allclose(found_updated, ref_updated,
                               rtol=rtol*100, atol=atol*100)
