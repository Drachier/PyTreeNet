"""
A module to provide various commonly used models for simulations.
"""
from typing import List, Tuple, Union, Dict

from numpy import eye, ndarray, zeros, mean

from .hamiltonian import Hamiltonian
from .tensorproduct import TensorProduct
from .sim_operators import (create_single_site_hamiltonian,
                            create_nearest_neighbour_hamiltonian,
                            single_site_operators)
from .common_operators import pauli_matrices
from ..core.tree_structure import TreeStructure

def ising_model(ref_tree: Union[TreeStructure, List[Tuple[str, str]]],
                ext_magn: float,
                factor: float = 1.0
                ) -> Hamiltonian:
    """
    Generates the Ising model with an external magnetic field for a full
    qubit tree, i.e. every node has a physical dimension of 2.

    Args:
        ref_tree (Union[TreeStructure, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeStructure
            object or as a list of tuples of nearest neighbours.
        ext_magn (float): The strength of the external magnetic field (Z-term).
        factor (float): The coupling factor between the nearest neighbours.
            (XX-term) Defaults to 1.0.
    
    Returns:
        Hamiltonian: The Hamiltonian of the Ising model.

    """
    paulis = pauli_matrices()
    return _abstract_ising_model(ref_tree, ext_magn, factor,
                                 ("Z", paulis[2]), ("X", paulis[0]))

def flipped_ising_model(ref_tree: Union[TreeStructure, List[Tuple[str, str]]],
                        ext_magn: float,
                        factor: float = 1.0
                        ) -> Hamiltonian:
    """
    Generates the Ising model with an external magnetic field for a full
    qubit tree, i.e. every node has a physical dimension of 2. The Ising model
    is flipped, i.e. X and Z operators are interchanged.

    Args:
        ref_tree (Union[TreeStructure, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeStructure
            object or as a list of tuples of nearest neighbours.
        ext_magn (float): The strength of the external magnetic field (X-term).
        factor (float): The coupling factor between the nearest neighbours.
            (ZZ-term) Defaults to 1.0.

    Returns:
        Hamiltonian: The Hamiltonian of the flipped Ising model.
    
    """
    paulis = pauli_matrices()
    return _abstract_ising_model(ref_tree, ext_magn, factor,
                                 ("X", paulis[0]), ("Z", paulis[2]))

def _abstract_ising_model(ref_tree: Union[TreeStructure, List[Tuple[str, str]]],
                          ext_magn: float,
                          factor: float,
                          ext_magn_op: Tuple[str,ndarray],
                          nn_op: Tuple[str,ndarray]
                          ) -> Hamiltonian:
    """
    Generates the Ising model with an external magnetic field for a full
    qubit tree, i.e. every node has a physical dimension of 2.

    Args:
        ref_tree (Union[TreeStructure, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeStructure
            object or as a list of tuples of nearest neighbours.
        ext_magn (float): The strength of the external magnetic field.
        factor (float): The coupling factor between the nearest neighbours.
        ext_magn_op (Tuple[str,ndarray]): The operator for the external
            magnetic field.
        nn_op (Tuple[str,ndarray]): The operator for the nearest neighbour
            coupling.

    Returns:
        Hamiltonian: The Hamiltonian of the Ising model.

    """
    local_dim = 2
    ext_magn_id = "mg" + ext_magn_op[0]
    single_dict = {ext_magn_id: -1*ext_magn*ext_magn_op[1]}
    # Produce the single site Hamiltonian
    # We need to prepare the identifiers for the single site Hamiltonian
    if isinstance(ref_tree, TreeStructure):
        single_site_structure = ref_tree
    else:
        # We assume all identifiers that possibly occur are in the nearest
        # neighbour list
        single_site_structure = [identifier
                                 for pair in ref_tree
                                 for identifier in pair]
        single_site_structure = list(set(single_site_structure))
    single_site_ham = create_single_site_hamiltonian(single_site_structure,
                                                     ext_magn_id,
                                                     conversion_dict=single_dict)
    # Produce the nearest neighbour Hamiltonian
    nn_minus_id = "m" + nn_op[0]
    nn_dict = {nn_minus_id: -1*factor*nn_op[1],
                nn_op[0]: nn_op[1]}
    nearest_neighbour_ham = create_nearest_neighbour_hamiltonian(ref_tree,
                                                                nn_minus_id,
                                                                nn_op[0],
                                                                conversion_dict=nn_dict)
    # Now we add all together into one Hamiltonian
    ham = Hamiltonian()
    ham.add_hamiltonian(single_site_ham)
    ham.add_hamiltonian(nearest_neighbour_ham)
    ham.conversion_dictionary[f"I{local_dim}"] = eye(local_dim)
    ham.conversion_dictionary["I1"] = eye(1)
    return ham

def local_magnetisation(structure: Union[TreeStructure,List[str]]
                        ) -> Dict[str,TensorProduct]:
    """
    Generates the local magnetisation operator for a given tree structure.

    Args:
        structure (Union[TreeStructure,List[str]]): The tree structure for
            which the local magnetisation operator should be generated. Can
            also be a list of node identifiers.
    
    Returns:
        Dict[str,TensorProduct]: The local magnetisation operators.

    """
    sigma_z = pauli_matrices()[2]
    return single_site_operators(sigma_z, structure)

def total_magnetisation(local_magnetisations: List[ndarray]
                        ) -> ndarray:
    """
    Computes the total magnetisation from the local magnetisations.
    
    Args:
        local_magnetisations (List[ndarray]): The local magnetisations as a
            list of arrays, where each array contains the local magnetisations
            for one site for different times.

    Returns:
        ndarray: The total magnetisation

        .. math::
            M = 1/L \sum_i^L m_i

    """
    if len(local_magnetisations) == 0:
        raise ValueError("No local magnetisations given.")
    num_sites = len(local_magnetisations)
    magn = zeros((num_sites, local_magnetisations[0].shape[0]),
                 dtype=local_magnetisations[0].dtype)
    for i in range(num_sites):
        magn[i] = local_magnetisations[i]
    return mean(magn, axis=0)
