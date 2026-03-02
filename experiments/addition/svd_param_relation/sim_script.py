"""
We want to compare the impact of the two different max bond dimensions
for the CBC and SRC addition.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import time
import os
from copy import deepcopy
from itertools import product

from h5py import File

from pytreenet.util.experiment_util.sim_params import SimulationParameters
from pytreenet.util.experiment_util.script_util import script_main
from pytreenet.core.addition.addition import AdditionMethod
from pytreenet.random.random_matrices import RandomDistribution
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.random.random_special_ttns import random_ttns
from pytreenet.time_evolution.results import Results
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.util.tensor_splitting import SVDParameters


@dataclass
class AdditionComparisonParams(SimulationParameters):
    """
    Parameters for the addition comparison simulation.
    """
    structure: TTNStructure = TTNStructure.MPS
    sys_size: int = 10
    phys_dim: int = 2
    init_bond_dim: int = 50
    max_bond_dim: int = 50
    min_bond_dim: int = 5
    step_bond_dim: int = 5
    addition_method: AdditionMethod = AdditionMethod.DIRECT
    num_additions: int = 2
    seed: int = 12334
    distr_low: float = -1.0
    distr_high: float = 1.0


RES_IDS = ("cache_bd", "add_bd", "error", "run_time")
RES_TYPES = (int, int, float, float)


def bond_dim_range(params: AdditionComparisonParams
                   ) -> range:
    """
    Generates a range of bond dimensions to be used in the simulation.

    Args:
        params (AdditionComparisonParams): The parameters for the simulation.

    Returns:
        range: A range of bond dimensions from min to max with the specified step.
    """
    return range(params.min_bond_dim,
                 params.max_bond_dim + 1,
                 params.step_bond_dim)

def init_results(params: AdditionComparisonParams) -> Results:
    """
    Initializes the results object for the simulation.

    Args:
        params (AdditionComparisonParams): The parameters for the simulation.

    Returns:
        Results: The initialized and empty results object.
    """
    # We have one result for each bond dimension
    num_res = len(list(product(bond_dim_range(params), repeat=2))) - 1
    results = Results()
    results.initialize(dict(zip(RES_IDS, RES_TYPES)),
                       num_res,
                       with_time=False)
    return results

def set_result_values(results: Results,
                      index: int,
                      cache_bd: int,
                      add_bd: int,
                      error: float,
                      run_time: float
                      ) -> None:
    """
    Sets the result values for a specific index in the results object.

    Args:
        results (Results): The results object.
        index (int): The index to set the values for.
        cache_bd (int): The cache bond dimension used for the TTNS.
        add_bd (int): The addition bond dimension used for the TTNS.
        error (float): The error compared to the scaled TTNS.
        run_time (float): The time taken to perform the addition.
    """
    res_values = (cache_bd, add_bd, error, run_time)
    for res_id, res_value in zip(RES_IDS, res_values):
        results.set_element(res_id, index, res_value)


def run_addition_comparison(params: AdditionComparisonParams) -> Results:
    """
    Runs the addition comparison simulation.

    This function creates a random TTNS with the initial bond dimension,
    then tests addition at varying bond dimensions by adding it to itself N times
    using the specified addition method, and compares the result to simply
    scaling the original TTNS by N.

    Args:
        params (AdditionComparisonParams): The parameters for the simulation.

    Returns:
        Results: The results of the simulation. It contains the
            bond dimension, error, and time taken to perform the
            additions for each bond dimension.
    """
    results = init_results(params)

    # Create random TTNSs once with initial bond dimension (outside loop)
    ttnss = [random_ttns(params.structure,
                       params.sys_size,
                       params.phys_dim,
                       params.init_bond_dim,
                       seed=params.seed,
                       distribution=RandomDistribution.UNIFORM,
                       low=params.distr_low,
                       high=params.distr_high
                       ) for _ in range(params.num_additions)]
    # Compute reference
    ref_add_method = AdditionMethod.DIRECT
    ref_add_fct = ref_add_method.get_function()
    ref_solution = ref_add_fct(deepcopy(ttnss))
    ref_solution = TreeTensorNetworkState.from_ttn(ref_solution)

    # Loop over bond dimensions
    for index, bond_dim in enumerate(product(bond_dim_range(params), repeat=2)):
        cache_bd = bond_dim[0]  # For SRC, we use the same bond dimension for the cache
        add_bd = bond_dim[1]
        # Perform the addition N times
        ttns_list = [deepcopy(ttns) for ttns in ttnss]

        # Time the addition operation
        start_time = time()

        # Add all TTNS in the list
        add_func = params.addition_method.get_function()
        svd_params = SVDParameters(max_bond_dim=add_bd)
        if params.addition_method is AdditionMethod.SRC:
            result_ttns = add_func(ttns_list,
                                   svd_params=svd_params,
                                   desired_dimension=cache_bd,
                                   seed=params.seed)
        elif params.addition_method is AdditionMethod.HALF_DENSITY_MATRIX:
            cache_svd_params = SVDParameters(max_bond_dim=cache_bd)
            result_ttns = add_func(ttns_list,
                                   svd_params=svd_params,
                                   cache_svd_params=cache_svd_params)
        else:
            errstr = f"Addition method {params.addition_method} is not supported for this simulation!"
            raise ValueError(errstr)

        end_time = time()

        # Compute error
        temp_ttns = TreeTensorNetworkState.from_ttn(result_ttns)
        error = temp_ttns.distance(ref_solution, normalise=True)

        # Store results
        set_result_values(results, index,
                          cache_bd, add_bd,
                          error, end_time - start_time)

    return results


def run_simulation(params: AdditionComparisonParams,
                   save_directory: str
                   ):
    """
    Runs the addition comparison simulation and saves the result.
    """
    results = run_addition_comparison(params)
    filename = params.get_hash() + ".h5"
    filepath = os.path.join(save_directory, filename)
    with File(filepath, "w") as f:
        results.save_to_h5(f)
        params.save_to_h5(f)


if __name__ == "__main__":
    script_main(run_simulation,
                AdditionComparisonParams)
