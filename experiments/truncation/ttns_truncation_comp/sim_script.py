"""
Implements the simulation script for the truncation comparison of different
trunctation methods for TTNS.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import time
import os
from copy import deepcopy

from h5py import File

from pytreenet.util.experiment_util.sim_params import SimulationParameters
from pytreenet.util.experiment_util.script_util import script_main
from pytreenet.core.truncation import (TruncationMethod,
                                       recursive_truncation,
                                       svd_truncation,
                                       single_site_fitting)
from pytreenet.random.random_matrices import RandomDistribution
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.random.random_special_ttns import random_ttns
from pytreenet.time_evolution.results import Results
from pytreenet.util.tensor_splitting import SVDParameters

@dataclass
class TruncationParams(SimulationParameters):
    """
    Parameters for the truncation comparison simulation.
    """
    structure: TTNStructure = TTNStructure.MPS
    sys_size: int = 10
    phys_dim: int = 2
    bond_dim: int = 10
    trunc_method: TruncationMethod = TruncationMethod.RECURSIVE
    random_trunc: bool = False
    max_target_bond_dim: int = 10
    min_target_bond_dim: int = 1
    step_target_bond_dim: int = 1
    seed: int = 12334
    distr_low: float = -1.0
    distr_high: float = 1.0

RES_IDS = ("bond_dim", "trunc_error", "run_time")

def init_results(params: TruncationParams) -> Results:
    """
    Initializes the results object for the simulation.

    Args:
        params (TruncationParams): The parameters for the simulation.

    Returns:
        Results: The initialized and empty results object.
    """
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
                      trunc_error: float,
                      run_time: float
                      ) -> None:
    """
    Sets the result values for a specific index in the results object.

    Args:
        results (Results): The results object.
        index (int): The index to set the values for.
        bond_dim (int): The bond dimension used in the truncation.
        trunc_error (float): The truncation error.
        run_time (float): The time taken to perform the truncation.
    """
    res_values = (bond_dim, trunc_error, run_time)
    for res_id, res_value in zip(RES_IDS, res_values):
        results.set_element(res_id, index, res_value)

def bond_dim_range(params: TruncationParams) -> range:
    """
    Generates a range of bond dimensions to be used in the simulation.

    Args:
        params (TruncationParams): The parameters for the simulation.

    Returns:
        range: A range of bond dimensions from min to max with the specified step.
    """
    return range(params.min_target_bond_dim,
                 params.max_target_bond_dim + 1,
                 params.step_target_bond_dim)

def run_fitting(params: TruncationParams) -> Results:
    """
    Runs the single-site fitting truncation method.

    Args:
        params (TruncationParams): The parameters for the simulation.
    
    Returns:
        Results: The results of the simulation. It contains the
            truncation error, the bond dimensions, and the
            time taken to perform the truncation for each dimension.
    """
    if params.random_trunc:
        raise NotImplementedError("Randomised truncation not implemented for "
                                  "single-site fitting!")
    ttns = random_ttns(params.structure,
                       params.sys_size,
                       params.phys_dim,
                       params.bond_dim,
                       seed=params.seed,
                       distribution=RandomDistribution.UNIFORM,
                       low=params.distr_low,
                       high=params.distr_high
                       )
    results = init_results(params)
    for index, bond_dim in enumerate(bond_dim_range(params)):
        init_ttns = random_ttns(params.structure,
                                 params.sys_size,
                                 params.phys_dim,
                                 bond_dim,
                                 seed=params.seed+1,
                                 distribution=RandomDistribution.UNIFORM,
                                 low=params.distr_low,
                                 high=params.distr_high
                                 )
        start_time = time()
        single_site_fitting(ttns, init_ttns,
                            2)
        end_time = time()
        trunc_error = ttns.distance(init_ttns, normalise=True)
        set_result_values(results, index,
                          bond_dim, trunc_error, end_time - start_time)
    return results

def run_svd_based_truncation(params: TruncationParams) -> Results:
    """
    Runs the SVD-based truncation method.

    Args:
        params (TruncationParams): The parameters for the simulation.

    Returns:
        Results: The results of the simulation. It contains the
            truncation error, the bond dimensions, and the
            time taken to perform the truncation for each dimension.
    """
    ttns = random_ttns(params.structure,
                       params.sys_size,
                       params.phys_dim,
                       params.bond_dim,
                       seed=params.seed,
                       distribution=RandomDistribution.UNIFORM,
                       low=params.distr_low,
                       high=params.distr_high)
    results = init_results(params)
    for index, bond_dim in enumerate(bond_dim_range(params)):
        comp_ttns = deepcopy(ttns)
        if params.trunc_method == TruncationMethod.RECURSIVE:
            trunc_func = recursive_truncation
        else:
            trunc_func = svd_truncation
        svd_params = SVDParameters(max_bond_dim=bond_dim,
                                   random=params.random_trunc)
        start_time = time()
        trunc_func(comp_ttns, svd_params)
        end_time = time()
        trunc_error = comp_ttns.distance(ttns, normalise=True)
        set_result_values(results, index,
                          bond_dim, trunc_error, end_time - start_time)
    return results

def run_simulation(params: TruncationParams,
                   save_directory: str
                   ):
    """
    Runs the truncation comparison simulation and saves the result.
    """
    if params.trunc_method == TruncationMethod.VARIATIONAL:
        results = run_fitting(params)
    else:
        results = run_svd_based_truncation(params)
    filename = params.get_hash() + ".h5"
    filepath = os.path.join(save_directory, filename)
    with File(filepath, "w") as f:
        results.save_to_h5(f)
        params.save_to_h5(f)

if __name__ == "__main__":
    script_main(run_simulation,
                TruncationParams)
