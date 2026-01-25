"""
The simulation script for the essentially all-to-all connectivity.
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from fractions import Fraction
import os

import numpy as np
from h5py import File

from pytreenet.ttno.ttno_class import TTNOFinder
from pytreenet.ttno.state_diagram import StateDiagram
from pytreenet.special_ttn.special_states import (TTNStructure,
                                                  STANDARD_NODE_PREFIX,
                                                  generate_zero_state,
                                                  Topology)
from pytreenet.util.experiment_util.sim_params import SimulationParameters
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.sim_operators import create_single_site_hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.util.experiment_util.script_util import script_main

class DimensionalityType(Enum):
    ONE_D = 1
    TWO_D = 2

    def num_sites(self, system_size: int) -> int:
        """
        Get the number of sites for the given dimensionality and system size.

        Args:
            system_size (int): The size of the system.
        
        Returns:
            int: The number of sites.
        """
        if self == DimensionalityType.ONE_D:
            return system_size
        elif self == DimensionalityType.TWO_D:
            return system_size ** 2
        else:
            raise ValueError(f"Unknown dimensionality: {self}!")

    def distance(self,
                 i: int,
                 j: int,
                 system_size: int
                 ) -> int:
        """
        Get the distance between two sites for the given dimensionality
        and system size.

        The distance is defined as the Manhattan distance.

        Args:
            i (int): The index of the first site.
            j (int): The index of the second site.
            system_size (int): The size of the system.
        
        Returns:
            int: The distance between the two sites.
        """
        if self == DimensionalityType.ONE_D:
            return abs(j - i)
        elif self == DimensionalityType.TWO_D:
            x1, y1 = divmod(i, system_size)
            x2, y2 = divmod(j, system_size)
            return abs(x2 - x1) + abs(y2 - y1)
        
    def topology(self) -> Topology:
        """
        Get the topology corresponding to the dimensionality.

        Returns:
            Topology: The corresponding topology.
        """
        if self == DimensionalityType.ONE_D:
            return Topology.CHAIN
        elif self == DimensionalityType.TWO_D:
            return Topology.SQUARE

@dataclass
class AllToAllSimParams(SimulationParameters):
    """
    The simulation parameters for the all-to-all comparison.
    """
    dimensionality: DimensionalityType = DimensionalityType.ONE_D
    system_size: int = 4
    ttn_structure: TTNStructure = TTNStructure.MPS
    method: TTNOFinder = TTNOFinder.SGE
    num_operators: int = 1

def generate_hamiltonian(systems_size: int,
                         dimensionality: DimensionalityType,
                         num_operators: int
                         ) -> Hamiltonian:
    """
    Generate a Hamiltonian for the all-to-all comparison.

    Args:
        systems_size (int): The size of the system.
        dimensionality (DimensionalityType): The dimensionality of the system.
        num_operators (int): The number of operators to include in the
            Hamiltonian.
    
    Returns:
        Hamiltonian: The generated Hamiltonian.
    """
    single_site_operator = "single_site_operator"
    single_site_prefactor = "single_site_prefactor"
    two_site_operators = [f"Operator{i}" for i in range(num_operators)]
    two_site_prefactors = [f"Prefactor{i}" for i in range(num_operators)]
    # Note that the actual value does not matter as we only construct the
    # State diagram not the actual TTNO.
    conv_dict = {op: np.zeros((2, 2), dtype=complex) for op in two_site_operators}
    conv_dict[single_site_operator] = np.zeros((2, 2), dtype=complex)
    coeffs_mapping = {fac: 2.0 for fac in two_site_prefactors}
    coeffs_mapping[single_site_prefactor] = 1.0
    num_sites = dimensionality.num_sites(systems_size)
    ham = Hamiltonian(coeffs_mapping=coeffs_mapping,
                      conversion_dictionary=conv_dict)
    # Add two site terms
    for i in range(num_sites - 1):
        for j in range(i+1, num_sites):
            for fac, op in zip(two_site_prefactors, two_site_operators):
                tp = TensorProduct()
                tp.add_operator(STANDARD_NODE_PREFIX + str(i), op)
                tp.add_operator(STANDARD_NODE_PREFIX + str(j), op)
                denominator = dimensionality.distance(i, j, systems_size)
                term = (Fraction(1, denominator), fac, tp)
                ham.add_term(term)
    # Add single site terms
    ham_ss = create_single_site_hamiltonian([STANDARD_NODE_PREFIX + str(i)
                                             for i in range(num_sites)],
                                             single_site_operator,
                                             (Fraction(1), single_site_prefactor))
    ham.add_hamiltonian(ham_ss)
    ham.include_identities([1,2])
    return ham

def get_size_of_state_diagram(params: AllToAllSimParams) -> int:
    """
    Get the size of the state diagram for the given simulation parameters.

    Args:
        params (AllToAllSimParams): The simulation parameters.
    
    Returns:
        int: The size of the state diagram.
    """
    ham = generate_hamiltonian(params.system_size,
                               params.dimensionality,
                               params.num_operators)
    tree_structure = generate_zero_state(params.system_size,
                                         params.ttn_structure,
                                         topology=params.dimensionality.topology())
    state_diagram = StateDiagram.from_hamiltonian(ham,
                                                  tree_structure,
                                                  method=params.method)
    return state_diagram.number_of_elements()

def run_and_save(dir_path: str,
                 params: AllToAllSimParams) -> None:
    """
    Run the simulation and save the results.

    Args:
        dir_path (str): The directory path to save the results.
        params (AllToAllSimParams): The simulation parameters.
    """
    size = get_size_of_state_diagram(params)
    param_hash = params.get_hash()
    file_path = os.path.join(dir_path, f"{param_hash}.h5")
    with File(file_path, "w") as f:
        params.save_to_h5(f)
        f.create_dataset("state_diagram_size", data=size)

if __name__ == "__main__":
    script_main(run_and_save, AllToAllSimParams)
