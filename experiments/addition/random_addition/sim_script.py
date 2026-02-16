"""
Implements the simulation script for comparing different addition methods
for adding multiple random TTNS together.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import time
import os
from copy import deepcopy

from h5py import File

from pytreenet.util.experiment_util.sim_params import SimulationParameters
from pytreenet.util.experiment_util.script_util import script_main
from pytreenet.core.addition.addition import AdditionMethod
from pytreenet.core.truncation import TruncationMethod
from pytreenet.random.random_matrices import RandomDistribution
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.random.random_special_ttns import random_ttns
from pytreenet.time_evolution.results import Results
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.util.tensor_splitting import SVDParameters

@dataclass
class RandomAdditionParams(SimulationParameters):
    """
    Parameters for the random addition comparison simulation.
    """
    structure: TTNStructure = TTNStructure.MPS
    sys_size: int = 10
    phys_dim: int = 2
    init_bond_dim: int = 4
    max_bond_dim: int = 40
    min_bond_dim: int = 4
    step_bond_dim: int = 4
    addition_method: AdditionMethod = AdditionMethod.DIRECT_TRUNCATE
    num_ttns: int = 2
    seed: int = 12334
    distr_low: float = -1.0
    distr_high: float = 1.0

RES_IDS = ("bond_dim", "error", "run_time")

def bond_dim_range(params: RandomAdditionParams) -> range:
    """
    Generates a range of bond dimensions to be used in the simulation.
    The maximum is num_ttns * init_bond_dim.

    Args:
        params (RandomAdditionParams): The parameters for the simulation.

    Returns:
        range: A range of bond dimensions from min to max with the specified step.
    """
    return range(params.min_bond_dim,
                 params.max_bond_dim + 1,
                 params.step_bond_dim)

def init_results(params: RandomAdditionParams) -> Results:
    """
    Initializes the results object for the simulation.

    Args:
        params (RandomAdditionParams): The parameters for the simulation.

    Returns:
        Results: The initialized and empty results object.
    """
    # We have one result for each bond dimension
    num_res = len(list(bond_dim_range(params))) - 1
    results = Results()
    res_dtypes = (int, float, float)
    results.initialize(dict(zip(RES_IDS, res_dtypes)),
                       num_res,
                       with_time=False)
    return results

def set_result_values(results: Results,
                      index: int,
                      bond_dim: int,
                      error: float,
                      run_time: float
                      ) -> None:
    """
    Sets the result values for a specific index in the results object.

    Args:
        results (Results): The results object.
        index (int): The index to set the values for.
        bond_dim (int): The bond dimension used for truncation.
        error (float): The error compared to the exact sum.
        run_time (float): The time taken to perform the addition.
    """
    res_values = (bond_dim, error, run_time)
    for res_id, res_value in zip(RES_IDS, res_values):
        results.set_element(res_id, index, res_value)

def run_random_addition_comparison(params: RandomAdditionParams) -> Results:
    """
    Runs the random addition comparison simulation.

    This function creates multiple random TTNS and tests addition at varying
    bond dimensions by adding them with truncation, and compares the result
    to the exact sum (without truncation).

    Args:
        params (RandomAdditionParams): The parameters for the simulation.

    Returns:
        Results: The results of the simulation. It contains the
            bond dimension, error, and time taken to perform the
            additions for each bond dimension.
    """
    results = init_results(params)

    # Create random TTNS once with initial bond dimension
    ttns_list = [random_ttns(params.structure,
                             params.sys_size,
                             params.phys_dim,
                             params.init_bond_dim,
                             seed=params.seed + i,
                             distribution=RandomDistribution.UNIFORM,
                             low=params.distr_low,
                             high=params.distr_high)
                 for i in range(params.num_ttns)]

    # Create exact reference using DIRECT addition (exact, no truncation)
    direct_add_func = AdditionMethod.DIRECT.get_function()
    reference_ttns_list = [deepcopy(ttns) for ttns in ttns_list]
    reference_ttns = direct_add_func(reference_ttns_list)

    # Get the actual addition function for testing
    add_func = params.addition_method.get_function()

    # Loop over bond dimensions for truncation
    for index, bond_dim in enumerate(bond_dim_range(params)):
        # Create fresh copies for each truncation test
        ttns_list_copy = [deepcopy(ttns) for ttns in ttns_list]

        # Time the addition operation
        start_time = time()

        # Add with truncation at this bond dimension
        svd_params = SVDParameters(max_bond_dim=bond_dim)
        if params.addition_method == AdditionMethod.DIRECT_TRUNCATE:
            result_ttns = add_func(ttns_list_copy, TruncationMethod.SVD,
                                   svd_params)
        elif params.addition_method in [AdditionMethod.DENSITY_MATRIX,
                                         AdditionMethod.HALF_DENSITY_MATRIX,
                                         AdditionMethod.SRC]:
            result_ttns = add_func(ttns_list_copy, svd_params=svd_params)
        else:
            result_ttns = add_func(ttns_list_copy)

        end_time = time()

        # Compute error
        temp_ttns = TreeTensorNetworkState.from_ttn(result_ttns)
        temp_ref = TreeTensorNetworkState.from_ttn(reference_ttns)
        error = temp_ttns.distance(temp_ref, normalise=True)

        # Store results
        set_result_values(results, index,
                          bond_dim, error, end_time - start_time)

    return results

def run_simulation(params: RandomAdditionParams,
                   save_directory: str
                   ):
    """
    Runs the random addition comparison simulation and saves the result.
    """
    results = run_random_addition_comparison(params)
    filename = params.get_hash() + ".h5"
    filepath = os.path.join(save_directory, filename)
    with File(filepath, "w") as f:
        results.save_to_h5(f)
        params.save_to_h5(f)

if __name__ == "__main__":
    script_main(run_simulation,
                RandomAdditionParams)
