"""
A module to provide various commonly used models for simulations.
"""

from numpy import eye

from .hamiltonian import Hamiltonian
from .sim_operators import (create_single_site_hamiltonian,
                            create_nearest_neighbour_hamiltonian)
from .common_operators import pauli_matrices
from ..core.tree_structure import TreeStructure

def ising_model(ref_tree: TreeStructure,
                ext_magn: float,
                factor: float = 1.0
                ) -> Hamiltonian:
    """
    Generates the Ising model with an external magnetic field.

    Args:
        ref_tree (TreeStructure): The reference tree, according to which the
            operators are combined.
        ext_magn (float): The strength of the external magnetic field.
        factor (float): The coupling factor between the nearest neighbours.
            Defaults to 1.0.
    
    Returns:
        Hamiltonian: The Hamiltonian of the Ising model.

    """
    local_dim = 2
    paulis = pauli_matrices()
    ham = Hamiltonian()
    single_dict = {"mgZ": -1*ext_magn*paulis[2]}
    single_site_ham = create_single_site_hamiltonian(ref_tree, "mgZ",
                                                    conversion_dict=single_dict)
    nn_dict = {"mX": -1*factor*paulis[0],
                "X": paulis[0]}
    nearest_neighbour_ham = create_nearest_neighbour_hamiltonian(ref_tree,
                                                                "mX",
                                                                "X",
                                                                conversion_dict=nn_dict)
    ham.add_hamiltonian(single_site_ham)
    ham.add_hamiltonian(nearest_neighbour_ham)
    ham.conversion_dictionary[f"I{local_dim}"] = eye(local_dim)
    return ham
