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
from pytreenet.ttns.ttns_ttno.application import (ApplicationMethod,
                                                  apply_ttno_to_ttns)
from pytreenet.random.random_matrices import RandomDistribution
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.random.random_special_ttns import random_ttns
from pytreenet.random.random_special_ttno import random_ttno
from pytreenet.time_evolution.results import Results
from pytreenet.util.tensor_splitting import SVDParameters

@dataclass
class ApplicationParams(SimulationParameters):
    """
    Parameters for the TTNO-TTNS application simulation.
    """
    structure: TTNStructure = TTNStructure.MPS
    sys_size: int = 10
    phys_dim: int = 2
    bond_dim: int = 10
    appl_method: ApplicationMethod = ApplicationMethod.DIRECT
    max_target_bond_dim: int = 10
    min_target_bond_dim: int = 1
    step_target_bond_dim: int = 1
    seed: int = 12334
    distr_low: float = -1.0
    distr_high: float = 1.0

RES_IDS = ("bond_dim", "trunc_error", "run_time")

def init_results(params: ApplicationParams) -> Results:
    """
    Initializes the results object for the simulation.

    Args:
        params (ApplicationParams): The parameters for the simulation.

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

def bond_dim_range(params: ApplicationParams) -> range:
    """
    Generates a range of bond dimensions to be used in the simulation.

    Args:
        params (ApplicationParams): The parameters for the simulation.

    Returns:
        range: A range of bond dimensions from min to max with the specified step.
    """
    return range(params.min_target_bond_dim,
                 params.max_target_bond_dim + 1,
                 params.step_target_bond_dim)

def run_simulation(params: ApplicationParams) -> Results:
    """
    Runs the TTNO-TTNS application simulation.
    """
    ttns = random_ttns(params.structure,
                       params.sys_size,
                       params.phys_dim,
                       params.bond_dim,
                       seed=params.seed,
                       distribution=RandomDistribution.UNIFORM,
                       low=params.distr_low,
                       high=params.distr_high)
    ttno = random_ttno(params.structure,
                       params.sys_size,
                       params.phys_dim,
                       params.bond_dim,
                       seed=params.seed,
                       distribution=RandomDistribution.UNIFORM,
                       low=params.distr_low,
                       high=params.distr_high)
    results = init_results(params)
    exact_ttns = apply_ttno_to_ttns(deepcopy(ttns),
                                    ttno,
                                    ApplicationMethod.DIRECT)
    for idx, bond_dim in enumerate(bond_dim_range(params)):
        svd_params = SVDParameters(max_bond_dim=bond_dim)
        if params.appl_method == ApplicationMethod.VARIATIONAL:
            kwargs = {"num_sweeps": 2,
                      "svd_params": svd_params}
        elif params.appl_method == ApplicationMethod.ZIPUP_VARIATIONAL:
            kwargs = {"num_sweeps": 2,
                      "var_svd_params": svd_params,
                      "dm_svd_params": svd_params}
        elif params.appl_method == ApplicationMethod.HALF_DENSITY_MATRIX_VARIATIONAL:
            kwargs = {"num_sweeps": 2,
                      "var_svd_params": svd_params,
                      "dm_svd_params": svd_params}
        elif params.appl_method == ApplicationMethod.SRC:
            kwargs = {"desired_dimension": bond_dim}
        else:
            kwargs = {"svd_params": svd_params}
        func = params.appl_method.get_function()
        start_time = time()
        approx_ttns = func(deepcopy(ttns),
                           deepcopy(ttno),
                           **kwargs)
        end_time = time()
        trunc_error = exact_ttns.distance(approx_ttns,
                                          normalise=True)
        set_result_values(results,
                          idx,
                          bond_dim,
                          trunc_error,
                          end_time - start_time)
    return results

def save_results(params: ApplicationParams,
                 results: Results,
                 save_directory: str
                 ) -> None:
    """
    Saves the simulation results to an HDF5 file.

    Args:
        params (ApplicationParams): The parameters for the simulation.
        results (Results): The results of the simulation.
        save_directory (str): The directory to save the results in.
    """
    filename = params.get_hash() + ".h5"
    filepath = os.path.join(save_directory, filename)
    with File(filepath, "w") as f:
        results.save_to_h5(f)
        params.save_to_h5(f)

def run_and_save(params: ApplicationParams,
                   save_directory: str
                   ):
    """
    Runs the truncation comparison simulation and saves the result.
    """
    results = run_simulation(params)
    save_results(params,
                 results,
                 save_directory)

if __name__ == "__main__":
    script_main(run_and_save,
                ApplicationParams)
