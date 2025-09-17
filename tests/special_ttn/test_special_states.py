"""
This module implements unittests for the special states in PyTreeNet.
"""
import pytest

import numpy as np

from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.special_ttn.special_states import (generate_constant_product_state,
                                                  generate_zero_state,
                                                  TTNStructure)
from pytreenet.operators.models.topology import Topology

@pytest.mark.parametrize("structure", [TTNStructure.MPS, TTNStructure.BINARY])
def test_generate_zero_state_1d(structure):
    """
    Test generating a zero state with value 0.
    """
    system_size = 5
    phys_dim = 2
    bond_dim = 2
    state = generate_zero_state(system_size,
                                 structure,
                                 bond_dim=bond_dim)
    total_dim = phys_dim ** system_size
    assert state.avg_bond_dim() == bond_dim
    found, _ = state.completely_contract_tree()
    found = found.reshape((total_dim,))
    correct = np.zeros((total_dim,), dtype=complex)
    correct[0] = 1.0
    np.testing.assert_allclose(found, correct)

def test_generate_zero_state_ttstar():
    """
    Test generating a zero state with T-structure.
    """
    system_size = 5
    phys_dim = 2
    bond_dim = 2
    state = generate_zero_state(system_size,
                                 TTNStructure.TSTAR,
                                 bond_dim=bond_dim,
                                 topology=Topology.TTOPOLOGY)
    total_dim = phys_dim ** (3 * system_size)
    assert state.avg_bond_dim() == bond_dim
    found, _ = state.completely_contract_tree()
    found = found.reshape((total_dim,))
    correct = np.zeros((total_dim,), dtype=complex)
    correct[0] = 1.0
    np.testing.assert_allclose(found, correct)

def test_generate_zero_state_exact():
    """
    Test generating a zero state with exact structure.
    """
    system_size = 5
    phys_dim = 2
    bond_dim = 2
    found = generate_zero_state(system_size,
                                TTNStructure.EXACT,
                                bond_dim=bond_dim)
    total_dim = phys_dim ** system_size
    correct = np.zeros((total_dim,), dtype=complex)
    correct[0] = 1.0
    np.testing.assert_allclose(found, correct)

@pytest.mark.parametrize("structure", [TTNStructure.MPS, TTNStructure.BINARY])
def test_generate_constant_product_state_1d(structure):
    """
    Test generating a constant product state with value 1 everywhere.
    """
    system_size = 5
    phys_dim = 2
    bond_dim = 3
    state = generate_constant_product_state(1,
                                            system_size,
                                            structure,
                                            phys_dim=phys_dim,
                                            bond_dim=bond_dim)
    total_dim = phys_dim ** system_size
    assert state.avg_bond_dim() == bond_dim
    found, _ = state.completely_contract_tree()
    found = found.reshape((total_dim,))
    correct = np.zeros((total_dim,), dtype=complex)
    correct[-1] = 1.0
    np.testing.assert_allclose(found, correct)

def test_generate_constant_product_state_ttstar():
    """
    Test generating a constant product state with T-structure.
    """
    system_size = 5
    phys_dim = 2
    bond_dim = 3
    state = generate_constant_product_state(1,
                                            system_size,
                                            TTNStructure.TSTAR,
                                            phys_dim=phys_dim,
                                            bond_dim=bond_dim,
                                            topology=Topology.TTOPOLOGY)
    total_dim = phys_dim ** (3 * system_size)
    assert state.avg_bond_dim() == bond_dim
    found, _ = state.completely_contract_tree()
    found = found.reshape((total_dim,))
    correct = np.zeros((total_dim,), dtype=complex)
    correct[-1] = 1.0
    np.testing.assert_allclose(found, correct)

def test_generate_constant_product_state_exact():
    """
    Test generating a constant product state with exact structure.
    """
    system_size = 5
    phys_dim = 2
    bond_dim = 3
    found = generate_constant_product_state(1,
                                            system_size,
                                            TTNStructure.EXACT,
                                            phys_dim=phys_dim,
                                            bond_dim=bond_dim)
    total_dim = phys_dim ** system_size
    correct = np.zeros((total_dim,), dtype=complex)
    correct[-1] = 1.0
    np.testing.assert_allclose(found, correct)
