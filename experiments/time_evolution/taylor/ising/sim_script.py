"""
Considers the the Taylor time evolution for a simple
Ising model with a transverse field.
"""
from __future__ import annotations
from dataclasses import dataclass
import os

from h5py import File

from pytreenet.time_evolution.application_based.taylor import Taylor
from pytreenet.time_evolution.results import Results
from pytreenet.core.addition.linear_combination import LinCombParams
from pytreenet.core.addition.addition import AdditionMethod
from pytreenet.ttns.ttns_ttno.application import ApplicationMethod
from pytreenet.operators.models.two_site_model import (IsingModel,
                                                       IsingParameters)
from pytreenet.util.tensor_splitting import SVDParameters
from pytreenet.ttno.ttno_class import TTNO
from pytreenet.special_ttn.special_states import (TTNStructure,
                                                  generate_zero_state,
                                                  STANDARD_NODE_PREFIX)
from pytreenet.operators.models.topology import Topology
from pytreenet.operators.models import local_magnetisation_from_topology
from pytreenet.util.experiment_util.script_util import script_main
from pytreenet.operators.exact_operators import (exact_local_magnetisation)
from pytreenet.time_evolution.exact_time_evolution import (ExactTimeEvolution)

@dataclass
class TaylorIsingParams(SVDParameters,
                     IsingParameters):
    """
    Parameters for the Taylor time evolution of the Ising model.
    """
    addition_method: AdditionMethod = AdditionMethod.HALF_DENSITY_MATRIX
    application_method: ApplicationMethod = ApplicationMethod.HALF_DENSITY_MATRIX
    topology: Topology = Topology.CHAIN
    ttn_structure: TTNStructure = TTNStructure.MPS
    system_size: int = 5
    time_step_size: float = 0.01
    final_time: float = 1.0
    order: int = 4

def run_Taylor(params: TaylorIsingParams) -> Results:
    """
    Runs the Taylor time evolution for the Ising model for the given parameters.
    """
    # Generate the initial state
    init_state = generate_zero_state(params.system_size,
                                     params.ttn_structure,
                                     topology=params.topology)
    # Generate the Hamiltonian
    hamiltonian = IsingModel.from_dataclass(params)
    hamiltonian = hamiltonian.generate_by_topology(params.topology,
                                                   params.system_size,
                                                   site_id_prefix=STANDARD_NODE_PREFIX)
    ham_ttno = TTNO.from_hamiltonian(hamiltonian,
                                     init_state)
    # Generate operators for exp value
    ops = local_magnetisation_from_topology(params.topology,
                                            params.system_size,
                                            site_prefix=STANDARD_NODE_PREFIX)
    # Create parameters for the linear combination of states during the time evolution
    if params.application_method == ApplicationMethod.SRC:
        kwargs_ap = {"desired_dimension": params.max_bond_dim}
    elif params.application_method in {ApplicationMethod.DENSITY_MATRIX,
                                       ApplicationMethod.HALF_DENSITY_MATRIX}:
        kwargs_ap = {"svd_params": params}
    else:
        errstr = f"Application method {params.application_method} not supported for linear combinations!"
        raise ValueError(errstr)
    lin_comb_params = LinCombParams(params.application_method,
                                    params.addition_method,
                                    kwargs_add={"svd_params": params},
                                    kwargs_ap=kwargs_ap)
    # Generate the time evolution class
    evo = Taylor(init_state,
              ham_ttno,
              params.time_step_size,
              params.final_time,
              ops,
              lin_comb_params=lin_comb_params)
    evo.run(pgbar=False)
    # Collect results
    return evo.results

def save_results(params: TaylorIsingParams,
                 results: Results,
                 dir_path: str):
    """
    Saves the results of the Taylor time evolution for the Ising model.

    The parameters are saved as metadata in the results file.

    Args:
        params (TaylorIsingParams): The parameters of the Taylor time evolution.
        results (Results): The results of the Taylor time evolution.
        dir_path (str): The directory path where the results should be saved.
    """
    os.makedirs(dir_path, exist_ok=True)
    param_hash = params.get_hash()
    file_path = os.path.join(dir_path,
                             f"{param_hash}.h5")
    with File(file_path, "w") as f:
        results.save_to_h5(f)
        params.save_to_h5(f)

def run_and_save(params: TaylorIsingParams,
                 dir_path: str):
    """
    Runs the Taylor time evolution for the Ising model and saves the results.

    Args:
        params (TaylorIsingParams): The parameters of the Taylor time evolution.
        dir_path (str): The directory path where the results should be saved.
    """
    results = run_Taylor(params)
    save_results(params, results, dir_path)
    exact_params = ExactParams.from_Taylor_params(params)
    exact_hash = exact_params.get_hash()
    exact_path = os.path.join(dir_path,
                                f"{exact_hash}.h5")
    if not os.path.exists(exact_path):
        exact_simulation(exact_params, dir_path)

@dataclass
class ExactParams(IsingParameters):
    """
    Parameters for the exact simulation of the Ising model.
    """
    topology: Topology = Topology.CHAIN
    system_size: int = 10
    time_step_size: float = 0.01
    final_time: float = 1.0

    @classmethod
    def from_Taylor_params(cls, Taylor_params: TaylorIsingParams) -> ExactParams:
        """
        Creates an ExactParams instance from an TaylorIsingParams instance.

        Args:
            Taylor_params (TaylorIsingParams): The TaylorIsingParams instance.
        
        Returns:
            ExactParams: The created ExactParams instance.
        """
        new = cls()
        for attr in vars(new).keys():
            if not hasattr(Taylor_params, attr):
                raise ValueError(f"TaylorIsingParams does not have attribute {attr}!")
            setattr(new, attr, getattr(Taylor_params, attr))
        return new

def exact_simulation(sim_params: ExactParams,
                     save_file_root: str
                     ) -> float:
    """
    Run an exact simulation of the Ising model with the given parameters.

    Args:
        sim_params (ExactParams): The simulation parameters.
        save_file_root (str): The root path for saving the results.

    Returns:
        float: The elapsed time for the simulation.
    """
    # Initialize the state
    state = generate_zero_state(sim_params.system_size,
                                TTNStructure.EXACT,
                                topology=sim_params.topology)
    # Generate the Hamiltonian
    ising_model = IsingModel.from_dataclass(sim_params)
    hamiltonian = ising_model.generate_matrix(sim_params.topology,
                                              sim_params.system_size)
    operators = exact_local_magnetisation([STANDARD_NODE_PREFIX + str(i)
                                           for i in range(sim_params.system_size)])
    # Create the time evolution algorithm
    time_evo_alg = ExactTimeEvolution(state,
                                      hamiltonian,
                                      sim_params.time_step_size,
                                      sim_params.final_time,
                                      operators)
    # Run the time evolution
    time_evo_alg.run(pgbar=False)
    param_hash = sim_params.get_hash()
    # Save the results
    save_file_path = os.path.join(save_file_root,
                                  f"{param_hash}.h5")
    with File(save_file_path, "w") as file:
        time_evo_alg.results.save_to_h5(file)
        sim_params.save_to_h5(file)

if __name__ == "__main__":
    script_main(run_and_save, TaylorIsingParams)
