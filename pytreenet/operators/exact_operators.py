"""
This module contains exact full matrix operators for many body systems.
"""
from typing import List, Dict

from numpy import ndarray, kron, eye, zeros, asarray

from .common_operators import pauli_matrices

def exact_ising_hamiltonian(coupling_strength: float,
                            g: float,
                            num_sites: int
                            ) -> ndarray:
    """
    Generate the exact Ising Hamiltonian for a chain of spins.

    Args:
        coupling_strength (float): The coupling strength.
        g (float): The magnetic field strength.
        num_sites (int): The number of sites in the chain.
    
    Returns:
        ndarray: The Hamiltonian as a matrix.
    """
    local_dim = 2
    hamiltonian = zeros((local_dim**num_sites, local_dim**num_sites),
                        dtype=complex)
    sigma_x, _, sigma_z = pauli_matrices()
    identity = eye(local_dim)
    # Two site terms
    for sitei in range(num_sites-1):
        term = 1
        for sitej in range(num_sites):
            if sitej == sitei:
                term = kron(term, -1 * coupling_strength * sigma_x)
            elif sitej == sitei+1:
                term = kron(term, sigma_x)
            else:
                term = kron(term, identity)
        hamiltonian += term
    # Single site terms
    for sitei in range(num_sites):
        term = exact_single_site_operator(-1 * g * sigma_z,
                                          sitei,
                                          num_sites)
        hamiltonian += term
    return hamiltonian

def exact_local_operators(site_ids: List[str],
                          local_operators: ndarray
                          ) -> Dict[str, ndarray]:
    """
    Generate the exact local operators for a chain of quantum systems.

    Each operator will be extended by identities to the full chain.

    Args:
        site_ids (List[str]): The site identifiers.
        local_operators (ndarray): The local operators.
    
    Returns:
        Dict[str,ndarray]: The local operators extended to the full chain.
            The dimensional order of the operators is the same as the order
            of the site identifiers.

    """
    num_sites = len(site_ids)
    operators = {}
    for site_index, site_id in enumerate(site_ids):
        operator = exact_single_site_operator(local_operators,
                                              site_index,
                                              num_sites)
        operators[site_id] = operator
    return operators

def exact_single_site_operator(local_operator: ndarray,
                               site_index: int,
                               num_sites: int
                               ) -> ndarray:
    """
    Generate the exact single site operator for a chain of quantum systems.

    This operator will be extended by identities to the full chain.

    Args:
        local_operator (ndarray): The local operator.
        site_index (int): The index of the site.
        num_sites (int): The number of sites.
    
    Returns:
        ndarray: The local operator extended to the full chain.

    """
    assert local_operator.shape[0] == local_operator.shape[1] \
        and local_operator.shape[0] == 2, \
        "**local_operator** must be a 2x2 matrix."
    local_dim = local_operator.shape[0]
    identity = eye(local_dim)
    operator = asarray([[1]])
    for site_index2 in range(num_sites):
        if site_index2 == site_index:
            operator = kron(operator, local_operator)
        else:
            operator = kron(operator, identity)
    return operator
