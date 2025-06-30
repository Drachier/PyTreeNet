"""
This script simulates the time evolution of a multi-site ising model
for different topologies.
"""
from __future__ import annotations
import sys
from enum import Enum
from dataclasses import dataclass
from fractions import Fraction
from time import time
import hashlib
import json
import os
import traceback

import numpy as np
from h5py import File

from pytreenet.operators.common_operators import pauli_matrices, ket_i
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.operators.sim_operators import (create_single_site_hamiltonian,
                                               single_site_operators)
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.special_ttn.binary import generate_binary_ttns
from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.special_ttn.star import StarTreeTensorState
from pytreenet.time_evolution.time_evolution import TimeEvoMode
from pytreenet.time_evolution.time_evo_enum import TimeEvoAlg
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator

NODE_PREFIX = "qubit"
CURRENT_PARAM_FILENAME = "current_parameters.json"
FILENAME_PREFIX = "simulation_"

class TTNStructure(Enum):
    """
    Enum for different tree tensor network structures.
    """
    MPS = "mps"
    BINARY = "binary"

    def color(self) -> str:
        """
        Return a color representation for the TTN structure.
        """
        if self == TTNStructure.MPS:
            return "green"
        elif self == TTNStructure.BINARY:
            return "blue"
        else:
            raise ValueError(f"Unknown TTN structure: {self.value}")
        
    def linestyle(self) -> str:
        """
        Return a linestyle representation for the TTN structure.
        """
        if self == TTNStructure.MPS:
            return "-"
        elif self == TTNStructure.BINARY:
            return "--"
        else:
            raise ValueError(f"Unknown TTN structure: {self.value}")

class Topology(Enum):
    """
    Enum for different topologies.
    """
    CHAIN = "chain"
    TTOPOLOGY = "t_topology"
    CAYLEY = "cayley"

@dataclass
class SimulationParameters:
    """
    Dataclass to hold simulation parameters.
    """
    ttns_structure: TTNStructure
    topology: Topology
    num_sites: int
    interaction_length: int
    strength: float
    init_bond_dim: int = 2

    def to_dict(self) -> dict:
        """
        Convert the simulation parameters to a dictionary.
        """
        return {
            "ttns_structure": self.ttns_structure.value,
            "topology": self.topology.value,
            "num_sites": self.num_sites,
            "interaction_length": self.interaction_length,
            "strength": self.strength,
            "init_bond_dim": self.init_bond_dim
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationParameters":
        """
        Create a SimulationParameters instance from a dictionary.
        """
        ttns_structure = TTNStructure(data["ttns_structure"])
        topology = Topology(data["topology"])
        return cls(
            ttns_structure,
            topology,
            data["num_sites"],
            data["interaction_length"],
            data["strength"],
            init_bond_dim=data.get("init_bond_dim", 2)
        )

    def save_to_h5(self, file: File):
        """
        Saves the simulation parameters to an HDF5 file.
        """
        group = file.create_group("simulation_parameters")
        group.attrs["ttns_structure"] = self.ttns_structure.value
        group.attrs["topology"] = self.topology.value
        group.attrs["num_sites"] = self.num_sites
        group.attrs["interaction_length"] = self.interaction_length
        group.attrs["strength"] = self.strength
        group.attrs["init_bond_dim"] = self.init_bond_dim

@dataclass
class TimeEvolutionParameters:
    """
    Dataclass to hold time evolution parameters.
    """
    time_evo_method: TimeEvoMode
    time_evo_algorithm: TimeEvoAlg
    time_step_size: float
    final_time: float
    ## For scipy methods
    atol: float = 1e-6
    rtol: float = 1e-6
    ## For 2TDVP and BUG
    max_bond_dim: int = 100
    rel_svalue: float = 1e-6
    abs_svalue: float = 1e-6

    def to_dict(self) -> dict:
        """
        Convert the time evolution parameters to a dictionary.
        """
        out = {
            "time_evo_method": self.time_evo_method.value,
            "time_evo_algorithm": self.time_evo_algorithm.value,
            "time_step_size": self.time_step_size,
            "final_time": self.final_time}
        if self.time_evo_method.is_scipy():
            out["atol"] = self.atol
            out["rtol"] = self.rtol
        if self.time_evo_algorithm.requires_svd():
            out["max_bond_dim"] = self.max_bond_dim
            out["rel_svalue"] = self.rel_svalue
            out["abs_svalue"] = self.abs_svalue
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "TimeEvolutionParameters":
        """
        Create a TimeEvolutionParameters instance from a dictionary.
        """
        time_evo_method = TimeEvoMode(data["time_evo_method"])
        time_evo_algorithm = TimeEvoAlg(data["time_evo_algorithm"])
        return cls(
            time_evo_method,
            time_evo_algorithm,
            data["time_step_size"],
            data["final_time"],
            atol=data.get("atol", 1e-6),
            rtol=data.get("rtol", 1e-6),
            max_bond_dim=data.get("max_bond_dim", 100),
            rel_svalue=data.get("rel_svalue", 1e-6),
            abs_svalue=data.get("abs_svalue", 1e-6)
        )

    def save_to_h5(self, file: File):
        """
        Saves the time evolution parameters to an HDF5 file.
        """
        group = file.create_group("time_evolution_parameters")
        group.attrs["time_evo_method"] = self.time_evo_method.value
        group.attrs["time_evo_algorithm"] = self.time_evo_algorithm.value
        if self.time_evo_method.is_scipy():
            group.attrs["atol"] = self.atol
            group.attrs["rtol"] = self.rtol
        if self.time_evo_algorithm.requires_svd():
            group.attrs["max_bond_dim"] = self.max_bond_dim
            group.attrs["rel_svalue"] = self.rel_svalue
            group.attrs["abs_svalue"] = self.abs_svalue

def initial_state(sim_params: SimulationParameters
                  ) -> TreeTensorNetworkState:
    """
    Generate the initial state based on the simulation parameters.
    """
    init_bond_dim = sim_params.init_bond_dim
    phys_dim = 2
    if sim_params.ttns_structure == TTNStructure.MPS and sim_params.topology == Topology.CHAIN:
        state = MatrixProductState.constant_product_state(0,phys_dim,
                                                          sim_params.num_sites,
                                                          node_prefix=NODE_PREFIX)
        state.pad_bond_dimensions(init_bond_dim)
        return state
    if sim_params.ttns_structure == TTNStructure.BINARY and sim_params.topology == Topology.CHAIN:
        local_state = ket_i(0, phys_dim)
        phys_tensor = np.zeros((init_bond_dim, phys_dim), dtype=complex)
        phys_tensor[0,:] = local_state
        state = generate_binary_ttns(sim_params.num_sites,
                                     init_bond_dim,
                                     phys_tensor,
                                     phys_prefix=NODE_PREFIX,
                                     )
        return state
    raise ValueError(f"Unsupported combination of structure {sim_params.ttns_structure} "
                     f"and topology {sim_params.topology}.")

def generate_ising_hamiltonian(sim_params: SimulationParameters
                                ) -> Hamiltonian:
    """
    Generate the Ising Hamiltonian based on the simulation parameters.

    The Hamiltonian is defined as:
    H = -J * sum_{i=1}^{N-1} otimes_{j=0}^{l-1} X_{i+j} - g * sum_{i=1}^{N} Z_i

    where J=1 is the interaction strength, g is the external field, and
    """
    ext_field = sim_params.strength
    hamiltonian = Hamiltonian()
    x, _, z = pauli_matrices()
    conv_dict = {"X": x, "Z": z}
    coeff_map = {"J": 1.0+0.0j, "g": complex(ext_field)}
    hamiltonian.conversion_dictionary = conv_dict
    hamiltonian.coeffs_mapping = coeff_map
    hamiltonian.include_identities([1,2])
    if sim_params.topology == Topology.CHAIN:
        qubits = [f"{NODE_PREFIX}{i}" for i in range(sim_params.num_sites)]
        single_site = create_single_site_hamiltonian(qubits,
                                                     "Z",
                                                     (Fraction(-1), "g")
                                                     )
        hamiltonian.add_hamiltonian(single_site)
        for i in range(sim_params.num_sites - sim_params.interaction_length):
            ops = {}
            for j in range(sim_params.interaction_length):
                ops[f"{NODE_PREFIX}{i + j}"] = "X"
            operator = TensorProduct(ops)
            hamiltonian.add_term((Fraction(-1), "J", operator))
        return hamiltonian
    raise ValueError(f"Unsupported topology {sim_params.topology}.")

def get_single_site_operators(length: int
                             ) -> dict[str, TensorProduct]:
    """
    Generate single-site operators for the given length.
    """
    ops = single_site_operators(pauli_matrices()[2],
                                [NODE_PREFIX + str(i) for i in range(length)],
                                with_factor=False)
    return ops

def get_param_hash(sim_params: SimulationParameters,
                     time_evo_params: TimeEvolutionParameters) -> str:
    """
    Generate a hash for the simulation parameters.
    """
    param_dict = sim_params.to_dict()
    param_dict.update(time_evo_params.to_dict())
    param_str = json.dumps(param_dict, sort_keys=True)
    hash = hashlib.sha256(param_str.encode()).hexdigest()[:30]
    return hash

def run_one_simulation(sim_params: SimulationParameters,
                       time_evo_params: TimeEvolutionParameters,
                       save_file_root: str
                       ) -> float:
    """
    Run a single simulation with the given parameters.
    
    Args:
        sim_params (SimulationParameters): The simulation parameters.
        time_evo_params (TimeEvolutionParameters): The time evolution
            parameters.
        save_file_root (str): The root path for saving the results.
    
    Returns:
        int: The simulation time.
    """
    # Initialize the state
    state = initial_state(sim_params)
    # Generate the Hamiltonian
    hamiltonian = generate_ising_hamiltonian(sim_params)
    ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, state)
    # Create operators to be evaluated during time evolution
    operators = get_single_site_operators(sim_params.num_sites)
    # Set up the time evolution algorithm
    if time_evo_params.time_evo_method.is_scipy():
        solver_options = {
            "atol": time_evo_params.atol,
            "rtol": time_evo_params.rtol
        }
    else:
        solver_options = None
    time_evo_alg_kind = time_evo_params.time_evo_algorithm
    config = time_evo_alg_kind.get_class().config_class()
    config.record_bond_dim = True
    config.record_norm = True
    config.record_max_bdim = True
    config.record_average_bdim = True
    config.record_total_size = True
    config.record_loschmidt_amplitude = True
    config.time_evo_mode = time_evo_params.time_evo_method
    if time_evo_alg_kind.requires_svd():
        config.max_bond_dim = time_evo_params.max_bond_dim
        config.rel_tol = time_evo_params.rel_svalue
        config.total_tol = time_evo_params.abs_svalue
    time_evo_alg = time_evo_alg_kind.get_algorithm_instance(state,
                                                            ttno,
                                                            time_evo_params.time_step_size,
                                                            time_evo_params.final_time,
                                                            operators,
                                                            config=config,
                                                            solver_options=solver_options)
    # Run the time evolution
    start_time = time()
    time_evo_alg.run(pgbar=False)
    end_time = time()
    elapsed_time = end_time - start_time
    param_hash = get_param_hash(sim_params, time_evo_params)
    # Save the results
    save_file_path = os.path.join(save_file_root, FILENAME_PREFIX + f"{param_hash}.h5")
    print(f"Saving results to {save_file_path}")
    with File(save_file_path, "w") as file:
        time_evo_alg.results.save_to_h5(file)
        sim_params.save_to_h5(file)
        time_evo_params.save_to_h5(file)
        file.attrs["elapsed_time"] = elapsed_time
    return elapsed_time

def load_params(path=CURRENT_PARAM_FILENAME
                ) -> tuple[SimulationParameters, TimeEvolutionParameters]:
    """
    Load the parameters from a JSON file.
    
    Args:
        path (str): The path to the JSON file containing the parameters.
    
    Returns:
        tuple: A tuple containing SimulationParameters and TimeEvolutionParameters.
    """
    with open(path, "r") as f:
        data = json.load(f)
    sim_params = SimulationParameters.from_dict(data)
    time_evo_params = TimeEvolutionParameters.from_dict(data)
    return sim_params, time_evo_params

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sim_script.py <save_directory>")
        sys.exit(1)

    try:
        save_directory = sys.argv[1]
        PARAM_PATH = os.path.join(save_directory, CURRENT_PARAM_FILENAME)
        SIM_PARAMS, TIME_EVO_PARAMS = load_params(PARAM_PATH)
        runtime = run_one_simulation(SIM_PARAMS, TIME_EVO_PARAMS, save_directory)
        print(f"Simulation completed in {runtime:.2f} seconds.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
