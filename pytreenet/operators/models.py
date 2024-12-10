"""
A module to provide various commonly used models for simulations.
"""
from typing import List, Tuple, Union, Dict

from numpy import eye

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
        ref_tree (nion[TreeStructure, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeStructure
            object or as a list of tuples of nearest neighbours.
        ext_magn (float): The strength of the external magnetic field.
        factor (float): The coupling factor between the nearest neighbours.
            Defaults to 1.0.
    
    Returns:
        Hamiltonian: The Hamiltonian of the Ising model.

    """
    local_dim = 2
    paulis = pauli_matrices()
    single_dict = {"mgZ": -1*ext_magn*paulis[2]}
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
                                                     "mgZ",
                                                     conversion_dict=single_dict)
    # Produce the nearest neighbour Hamiltonian
    nn_dict = {"mX": -1*factor*paulis[0],
                "X": paulis[0]}
    nearest_neighbour_ham = create_nearest_neighbour_hamiltonian(ref_tree,
                                                                "mX",
                                                                "X",
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
