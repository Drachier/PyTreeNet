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
import io
import contextlib
from typing import Union, Optional
from h5py import File
from numpy import ndarray, complex128, array

from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.operators.sim_operators import single_site_operators
from pytreenet.time_evolution.time_evo_enum import TimeEvoAlg
from pytreenet.ttns.ttndo import (mps_ttndo_for_product_state,
                                    binary_ttndo_for_product_state,
                                    symmetric_ttndo_for_product_state,
                                    SymmetricTTNDO, BINARYTTNDO,
                                    PHYS_PREFIX)
from pytreenet.operators.models import (ising_model,
                                        flipped_ising_model,
                                        local_magnetisation)
from pytreenet.operators.lindbladian import generate_lindbladian
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator
from pytreenet.operators.common_operators import bosonic_operators
from pytreenet.time_evolution.time_evolution import TimeEvoMode
from pytreenet.util.tensor_splitting import TruncationLevel


CURRENT_PARAM_FILENAME = "current_parameters.json"

# Global initial quantum state
INITIAL_STATE_ZERO = array([1.0, 0.0], dtype=complex128)  # |0⟩ state


class TimeStepLevel(Enum):
    """Labels representing grid fineness levels for the time step."""
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"

class TTNStructure(Enum):
    """
    Enum for different tree tensor network structures.
    """
    MPS = "mps"
    BINARY = "binary"
    SYMMETRIC = "symmetric"


@dataclass
class SimulationParameters:
    """
    Dataclass to hold simulation parameters.
    """
    ttns_structure: TTNStructure
    num_sites: int
    coupling: float
    ext_magn: float
    relaxation_rate: float
    dephasing_rate: float
    init_bond_dim: int
    depth: Optional[int]

    def to_dict(self) -> dict:
        """
        Convert the simulation parameters to a dictionary.
        """
        return {"ttns_structure": self.ttns_structure.value,
                "num_sites": self.num_sites,
                "coupling": self.coupling,
                "ext_magn": self.ext_magn,
                "relaxation_rate": self.relaxation_rate,
                "dephasing_rate": self.dephasing_rate,
                "init_bond_dim": self.init_bond_dim,
                "depth": self.depth}

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationParameters":
        """
        Create a SimulationParameters instance from a dictionary.
        """
        ttns_structure = TTNStructure(data["ttns_structure"])
        return cls(
            ttns_structure,
            data["num_sites"],
            data["coupling"],
            data["ext_magn"],
            data["relaxation_rate"],
            data["dephasing_rate"],
            init_bond_dim=data.get("init_bond_dim", 2),
            depth=data.get("depth"))

    def save_to_h5(self, file: File):
        """
        Saves the simulation parameters to an HDF5 file.
        """
        group = file.create_group("simulation_parameters")
        group.attrs["ttns_structure"] = self.ttns_structure.value
        group.attrs["num_sites"] = self.num_sites
        group.attrs["coupling"] = self.coupling
        group.attrs["ext_magn"] = self.ext_magn
        group.attrs["relaxation_rate"] = self.relaxation_rate
        group.attrs["dephasing_rate"] = self.dephasing_rate
        group.attrs["init_bond_dim"] = self.init_bond_dim
        # Handle None values that HDF5 can't serialize
        group.attrs["depth"] = self.depth if self.depth is not None else -1

@dataclass
class TimeEvolutionParameters:
    """
    Dataclass to hold time evolution parameters.
    """
    time_evo_method: TimeEvoMode
    time_evo_algorithm: TimeEvoAlg
    time_step_size: float
    evaluation_time: int
    final_time: float

    atol: float
    rtol: float

    # Optional label for time-step fineness
    time_step_level: Optional[TimeStepLevel] = None

    max_bond_dim: int = 100
    rel_svalue: float = 1e-6
    abs_svalue: float = 1e-6
    renorm: bool = False
    sum_trunc: bool = False
    sum_renorm: bool = True
    truncation_level: Optional[TruncationLevel] = None

    def to_dict(self) -> dict:
        """
        Convert the time evolution parameters to a dictionary.
        """
        out = {
            "time_evo_method": self.time_evo_method.value,
            "time_evo_algorithm": self.time_evo_algorithm.value,
            "time_step_size": self.time_step_size,
            "evaluation_time": self.evaluation_time,
            "time_step_level": self.time_step_level.value if self.time_step_level else None,
            "final_time": self.final_time,
            "truncation_level": self.truncation_level.value if self.truncation_level else None}
        if self.time_evo_method.is_scipy():
            out["atol"] = self.atol
            out["rtol"] = self.rtol
        if self.time_evo_algorithm.requires_svd():
            out["max_bond_dim"] = self.max_bond_dim
            out["rel_svalue"] = self.rel_svalue
            out["abs_svalue"] = self.abs_svalue
            out["renorm"] = self.renorm
            out["sum_trunc"] = self.sum_trunc
            out["sum_renorm"] = self.sum_renorm
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "TimeEvolutionParameters":
        """
        Create a TimeEvolutionParameters instance from a dictionary.
        """
        time_evo_method = TimeEvoMode(data["time_evo_method"])
        time_evo_algorithm = TimeEvoAlg(data["time_evo_algorithm"])
        truncation_level = TruncationLevel(data["truncation_level"]) if data.get("truncation_level") else None
        # Extract time_step_level if present
        tsl = data.get("time_step_level")
        time_step_level = TimeStepLevel(tsl) if tsl else None
        return cls(time_evo_method,
                    time_evo_algorithm,
                    data["time_step_size"],
                    data["evaluation_time"],
                    final_time=data["final_time"],
                    time_step_level=time_step_level,
                    atol=data.get("atol", 1e-6),
                    rtol=data.get("rtol", 1e-6),
                    max_bond_dim=data.get("max_bond_dim", 100),
                    rel_svalue=data.get("rel_svalue", 1e-6),
                    abs_svalue=data.get("abs_svalue", 1e-6),
                    renorm=data.get("renorm", False),
                    sum_trunc=data.get("sum_trunc", False),
                    sum_renorm=data.get("sum_renorm", True),
                    truncation_level=truncation_level)

    def save_to_h5(self, file: File):
        """
        Saves the time evolution parameters to an HDF5 file.
        """
        group = file.create_group("time_evolution_parameters")
        group.attrs["time_evo_method"] = self.time_evo_method.value
        group.attrs["time_evo_algorithm"] = self.time_evo_algorithm.value
        group.attrs["truncation_level"] = self.truncation_level.value if self.truncation_level else ""
        group.attrs["time_step_level"] = self.time_step_level.value if self.time_step_level else ""
        if self.time_evo_method.is_scipy():
            group.attrs["atol"] = self.atol
            group.attrs["rtol"] = self.rtol
        if self.time_evo_algorithm.requires_svd():
            group.attrs["max_bond_dim"] = self.max_bond_dim
            group.attrs["rel_svalue"] = self.rel_svalue
            group.attrs["abs_svalue"] = self.abs_svalue
            group.attrs["renorm"] = self.renorm
            group.attrs["sum_trunc"] = self.sum_trunc
            group.attrs["sum_renorm"] = self.sum_renorm


def initial_ttndo(sim_params: SimulationParameters):
    """
    Generate the initial TTNDO state based on the simulation parameters.
    
    Returns the appropriate TTNDO structure for time evolution algorithms.
    """
    init_bond_dim = sim_params.init_bond_dim
    depth = sim_params.depth

    # Use the global constant for the initial state |0⟩
    phys_tensor = INITIAL_STATE_ZERO

    if sim_params.ttns_structure == TTNStructure.MPS:
        ttns, ttndo = mps_ttndo_for_product_state(
            num_phys=sim_params.num_sites,
            bond_dim=init_bond_dim,
            phys_tensor=phys_tensor)
        return ttns, ttndo

    elif sim_params.ttns_structure == TTNStructure.BINARY:
        ttns, ttndo = binary_ttndo_for_product_state(
            num_phys=sim_params.num_sites,
            bond_dim=init_bond_dim,
            phys_tensor=phys_tensor,
            depth=depth)
        return ttns, ttndo

    elif sim_params.ttns_structure == TTNStructure.SYMMETRIC:
        ttns, ttndo = symmetric_ttndo_for_product_state(
            num_phys=sim_params.num_sites,
            bond_dim=init_bond_dim,
            phys_tensor=phys_tensor,
            depth=depth,
            root_bond_dim=init_bond_dim)
        # Symmetric function only returns ttndo, so we return None for ttns
        return ttns, ttndo

    raise ValueError(f"Unsupported combination of structure {sim_params.ttns_structure}")

def open_ising_model_ttno(sim_params: SimulationParameters,
                          ttndo: Union[SymmetricTTNDO, BINARYTTNDO],
                          flipped: bool = False
                          ) -> tuple[TreeTensorNetworkOperator, Hamiltonian]:
    """
    Generates the Hamiltonian for the open Ising model.

    Additionally to the usual TFI model two jump operators act on every site
    separately. The first jump operator is relaxation and the second is 
    dephasing.

    Args:
        sim_params. (int): The sim_params. of the system.
        ttndo (SymmetricTTNDO): The TTNDO of the system.
        flipped (bool): Whether to use the flipped Ising model.

    Returns:
        TreeTensorNetworkOperator: The TTNO.
    
    """
    node_identifiers = [f"{PHYS_PREFIX}{i}" for i in range(sim_params.num_sites)]
    nn_pairs = [(node_identifiers[i], node_identifiers[i+1])
                for i in range(sim_params.num_sites-1)]
    if flipped:
        ham = flipped_ising_model(nn_pairs,
                                  sim_params.ext_magn,
                                  sim_params.coupling)
    else:
        ham = ising_model(nn_pairs,
                          sim_params.ext_magn,
                          sim_params.coupling)
    jump_ops, conversion_dict, coeff_dict = jump_operators(sim_params)
    lindbladian = generate_lindbladian(ham,
                                        jump_ops,
                                        conversion_dict,
                                        coeff_dict)
    ttno = TreeTensorNetworkOperator.from_hamiltonian(lindbladian,
                                                      ttndo)
    return ttno, ham

def jump_operators(sim_params: SimulationParameters,
                   ) -> tuple[list[tuple[Fraction,str,TensorProduct]],
                              dict[str,ndarray], dict[str,complex]]:
    """
    Generates the jump operators for the open Ising model.

    Args:
        length (int): The length of the system.
        relaxation_rate (float): The relaxation rate.
        dephasing_rate (float): The dephasing rate.

    Returns:
        list[tuple[Fraction,str,TensorProduct]]: The jump operator terms.
        dict[str,ndarray]: The jump operator conversion dictionary.
        dict[str,float]: The jump operator coefficient mapping
    
    """
    node_identifiers = [f"{PHYS_PREFIX}{i}" for i in range(sim_params.num_sites)]
    # Create relaxation jump operators
    rel_fac_name = "relaxation_rate"
    factor = (Fraction(1), rel_fac_name)
    relax_name = "sigma_-"
    relaxation_ops = single_site_operators(relax_name,
                                           node_identifiers,
                                           factor)
    # Create dephasing jump operators
    deph_fac_name = "dephasing_rate"
    factor = (Fraction(1), deph_fac_name)
    deph_name = "Z"
    dephasing_ops = single_site_operators(deph_name,
                                          node_identifiers,
                                          factor)
    # Combine the jump operators
    jump_ops = list(relaxation_ops.values())
    jump_ops.extend(list(dephasing_ops.values()))
    # Create the jump operator matrix mapping
    conversion_dict = {relax_name: bosonic_operators()[1],
                       deph_name: pauli_matrices()[2]}
    # Create the jump operator coefficient mapping
    coeff_dict = {rel_fac_name: complex(sim_params.relaxation_rate),
                  deph_fac_name: complex(sim_params.dephasing_rate)}
    return jump_ops, conversion_dict, coeff_dict

def open_ising_operators(sim_params: SimulationParameters,
                         ising_ham: Hamiltonian
                         ) -> dict:
    """
    Generates the operators to be evaluated for the open Ising model.
    
    Args:
        length (int): The length of the system.
        ttns (TreeTensorNetworkState): The TTN state of the system.
        ising_ham (Hamiltonian): The Ising Hamiltonian.
    
    Returns:
        dict: The operators.

    """
    ttns , _ = initial_ttndo(sim_params)
    node_identifiers = [f"{PHYS_PREFIX}{i}" for i in range(sim_params.num_sites)]
    ops = {key: tup[2]
           for key, tup in local_magnetisation(node_identifiers).items()}
    ttno = TreeTensorNetworkOperator.from_hamiltonian(ising_ham,
                                                      ttns)
    ops["energy"] = ttno
    return ops

def get_param_hash(sim_params: SimulationParameters,
                   time_evo_params: TimeEvolutionParameters) -> str:
    """
    Generate a hash for the simulation parameters.
    """
    param_dict = sim_params.to_dict()
    time_dict = time_evo_params.to_dict()

    param_dict.update(time_dict)
    param_str = json.dumps(param_dict, sort_keys=True)
    param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:30]
    return param_hash

def run_one_simulation(sim_params: SimulationParameters,
                       time_evo_params: TimeEvolutionParameters,
                       save_file_root: str
                       ) -> float:

    _ , ttndo = initial_ttndo(sim_params)
    lindbladian_ttno, ising_ham = open_ising_model_ttno(sim_params,
                                            ttndo)
    operators = open_ising_operators(sim_params,
                                     ising_ham)

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
    config.record_loschmidt_amplitude = False
    config.time_evo_mode = time_evo_params.time_evo_method

    if time_evo_alg_kind.requires_svd():
        config.max_bond_dim = time_evo_params.max_bond_dim
        config.rel_tol = time_evo_params.rel_svalue
        config.total_tol = time_evo_params.abs_svalue
        config.renorm = time_evo_params.renorm
        config.sum_trunc = time_evo_params.sum_trunc
        config.sum_renorm = time_evo_params.sum_renorm

    time_evo_alg = time_evo_alg_kind.get_algorithm_instance(ttndo,
                                                            lindbladian_ttno,
                                                            time_evo_params.time_step_size,
                                                            time_evo_params.final_time,
                                                            operators,
                                                            config=config,
                                                            solver_options=solver_options)

    # Capture stdout and stderr to log what would normally be printed
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Run the time evolution
    start_time = time()
    with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
        time_evo_alg.run(time_evo_params.evaluation_time, pgbar=False)


    end_time = time()
    elapsed_time = end_time - start_time
    param_hash = get_param_hash(sim_params, time_evo_params)
    # Save the results
    save_file_path = os.path.join(save_file_root, f"simulation_{param_hash}.h5")
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
    with open(path, "r", encoding="utf-8") as f:
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
