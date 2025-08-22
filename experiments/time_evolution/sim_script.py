"""
This script simulates the time evolution of a multi-site ising model
for different topologies.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import time
import os

from h5py import File

from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.operators.models import local_magnetisation
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.time_evolution.time_evolution import TimeEvoMode, TimeEvolution
from pytreenet.time_evolution.time_evo_enum import TimeEvoAlg
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator
from pytreenet.operators.exact_operators import (exact_zero_state,
                                                 exact_ising_hamiltonian,
                                                 exact_local_magnetisation)
from pytreenet.time_evolution.exact_time_evolution import (ExactTimeEvolution)
from pytreenet.operators.models.two_site_model import (IsingModel,
                                                       IsingParameters)
from pytreenet.special_ttn.special_states import (TTNStructure,
                                                  generate_zero_state)
from pytreenet.operators.models.topology import Topology
from pytreenet.util.experiment_util.script_util import script_main

NODE_PREFIX = "qubit"
FILENAME_PREFIX = "simulation_"

@dataclass
class LocalSimulationParameters(IsingParameters):
    """
    Dataclass to hold simulation parameters.

    Attributes:
        ttns_structure (TTNSTructure): The structure of the TTNS used for 
            simulation.
        topology (Topology): The system topology defining the Hamiltonian
            used for simulation.
        system_size (int): The characteristic size of the system. For a chain
            this is the toal number of sites, for a star this is the chain
            length, for a Cayley tree this is the depth.
    """
    ttns_structure: TTNStructure = TTNStructure.MPS
    topology: Topology = Topology.CHAIN
    system_size: int = 1
    init_bond_dim: int = 2

@dataclass
class TimeEvolutionParameters:
    """
    Dataclass to hold time evolution parameters.
    """
    time_evo_method: TimeEvoMode = TimeEvoMode.CHEBYSHEV
    time_evo_algorithm: TimeEvoAlg = TimeEvoAlg.BUG
    time_step_size: float = 0.01
    final_time: float = 1.0
    ## For scipy methods
    atol: float = 1e-6
    rtol: float = 1e-6
    ## For 2TDVP and BUG
    max_bond_dim: int = 100
    rel_svalue: float = 1e-6
    abs_svalue: float = 1e-6

@dataclass
class TotalParameters(LocalSimulationParameters,
                      TimeEvolutionParameters):
    """
    A dataclass to hold all parameters for the simulation.
    """

def initial_state(sim_params: LocalSimulationParameters
                  ) -> TreeTensorNetworkState:
    """
    Generate the initial state based on the simulation parameters.
    """
    phys_dim = 2
    oned_structures = [TTNStructure.MPS,
                       TTNStructure.BINARY]
    if sim_params.topology == Topology.CHAIN and sim_params.ttns_structure in oned_structures:
        system_size = sim_params.system_size
    elif sim_params.topology == Topology.TTOPOLOGY and sim_params.ttns_structure in oned_structures:
        system_size = 3 * sim_params.system_size
    elif sim_params.topology == Topology.TTOPOLOGY and sim_params.ttns_structure == TTNStructure.TSTAR:
        system_size = sim_params.system_size
    else:
        errstr = f"Unsupported combination of structure {sim_params.ttns_structure} "
        errstr += f"and topology {sim_params.topology}."
        raise ValueError(errstr)
    state = generate_zero_state(system_size,
                                sim_params.ttns_structure,
                                phys_dim=phys_dim,
                                node_prefix=NODE_PREFIX,
                                bond_dim=sim_params.init_bond_dim)
    assert isinstance(state, TreeTensorNetworkState)
    return state

def generate_ising_hamiltonian(sim_params: LocalSimulationParameters
                                ) -> Hamiltonian:
    """
    Generate the Ising Hamiltonian based on the simulation parameters.

    The Hamiltonian is defined as:
    H = -J * sum_{i=1}^{N-1} otimes_{j=0}^{l-1} X_{i+j} - g * sum_{i=1}^{N} Z_i

    where J=1 is the interaction strength, g is the external field, and
    """
    model = IsingModel.from_dataclass(sim_params)
    return model.generate_by_topology(sim_params.topology,
                                      sim_params.system_size,
                                      site_id_prefix=NODE_PREFIX)

def get_single_site_operators(length: int
                             ) -> dict[str, TensorProduct]:
    """
    Generate single-site operators for the given length.
    """
    ops = local_magnetisation([NODE_PREFIX + str(i) for i in range(length)])
    return ops

def set_up_time_evolution(sim_params: TimeEvolutionParameters,
                          state: TreeTensorNetworkState,
                          ttno: TreeTensorNetworkOperator,
                          operators: dict[str, TensorProduct]
                          ) -> TimeEvolution:
    """
    Set up the time evolution algorithm based on the simulation parameters.

    Args:
        sim_params (TimeEvolutionParameters): The time evolution parameters.
        state (TreeTensorNetworkState): The initial state of the system.
        ttno (TreeTensorNetworkOperator): The operator representing the
            Hamiltonian of the system.
        operators (dict[str, TensorProduct]): The operators to be evaluated
            during time evolution.
    
    Returns:
        TimeEvolution: An instance of the time evolution algorithm.
    """
    if sim_params.time_evo_method.is_scipy():
        solver_options = {
            "atol": sim_params.atol,
            "rtol": sim_params.rtol
        }
    else:
        solver_options = None
    time_evo_alg_kind = sim_params.time_evo_algorithm
    config = time_evo_alg_kind.get_class().config_class()
    config.record_bond_dim = True
    config.record_norm = True
    config.record_max_bdim = True
    config.record_average_bdim = True
    config.record_total_size = True
    config.record_loschmidt_amplitude = True
    config.time_evo_mode = sim_params.time_evo_method
    if time_evo_alg_kind.requires_svd():
        config.max_bond_dim = sim_params.max_bond_dim
        config.rel_tol = sim_params.rel_svalue
        config.total_tol = sim_params.abs_svalue
    time_evo = time_evo_alg_kind.get_algorithm_instance(state,
                                                        ttno,
                                                        sim_params.time_step_size,
                                                        sim_params.final_time,
                                                        operators,
                                                        config=config,
                                                        solver_options=solver_options)
    return time_evo

def exact_simulation(sim_params: LocalSimulationParameters,
                     save_file_root: str
                     ) -> float:
    """
    Run an exact simulation of the Ising model with the given parameters.

    Args:
        sim_params (LocalSimulationParameters): The simulation parameters.
        save_file_root (str): The root path for saving the results.

    Returns:
        float: The elapsed time for the simulation.
    """
    # Initialize the state
    state = exact_zero_state(sim_params.system_size)
    # Generate the Hamiltonian
    hamiltonian = exact_ising_hamiltonian(sim_params.factor,
                                          sim_params.ext_magn,
                                          sim_params.interaction_range)
    operators = exact_local_magnetisation([NODE_PREFIX + str(i) for i in range(sim_params.system_size)])
    # Create the time evolution algorithm
    time_evo_alg = ExactTimeEvolution(state,
                                      hamiltonian,
                                      sim_params.time_step_size,
                                      sim_params.final_time,
                                      operators)
    # Run the time evolution
    start_time = time()
    time_evo_alg.run(pgbar=False)
    end_time = time()
    elapsed_time = end_time - start_time
    param_hash = sim_params.get_hash()
    # Save the results
    save_file_path = os.path.join(save_file_root, FILENAME_PREFIX + f"{param_hash}.h5")
    print(f"Saving results to {save_file_path}")
    with File(save_file_path, "w") as file:
        time_evo_alg.results.save_to_h5(file)
        sim_params.save_to_h5(file)
        file.attrs["elapsed_time"] = elapsed_time
    return elapsed_time

def run_ttn_simulation(sim_params: TotalParameters,
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
    operators = get_single_site_operators(sim_params.system_size)
    # Set up the time evolution algorithm
    time_evo_alg = set_up_time_evolution(sim_params, state, ttno, operators)
    # Run the time evolution
    start_time = time()
    time_evo_alg.run(pgbar=False)
    end_time = time()
    elapsed_time = end_time - start_time
    param_hash = sim_params.get_hash()
    # Save the results
    save_file_path = os.path.join(save_file_root, FILENAME_PREFIX + f"{param_hash}.h5")
    print(f"Saving results to {save_file_path}")
    with File(save_file_path, "w") as file:
        time_evo_alg.results.save_to_h5(file)
        sim_params.save_to_h5(file)
        file.attrs["elapsed_time"] = elapsed_time
    return elapsed_time

def run_one_simulation(sim_params: TotalParameters,
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
        float: The simulation time.
    """
    if sim_params.time_evo_algorithm == TimeEvoAlg.EXACT:
        return exact_simulation(sim_params, save_file_root)
    else:
        return run_ttn_simulation(sim_params, save_file_root)

if __name__ == "__main__":
    script_main(run_one_simulation,
                TotalParameters)
