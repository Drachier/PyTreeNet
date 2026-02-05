"""
The simulation script to add a TTNS to itself.
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
                                       truncate_ttns)
from pytreenet.random.random_matrices import RandomDistribution
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.random.random_special_ttns import random_ttns
from pytreenet.time_evolution.results import Results
from pytreenet.util.tensor_splitting import SVDParameters
from pytreenet.core.addition.addition import (AdditionMethod,
                                              add_ttns)
from pytreenet.ttns.ttns import TreeTensorNetworkState

RES_IDS = ("bond_dim", "trunc_error", "run_time")

@dataclass
class AdditionParams(SimulationParameters):
    """
    Parameters for the TTNS addition simulation.
    """
    structure: TTNStructure = TTNStructure.MPS
    sys_size: int = 10
    phys_dim: int = 2
    bond_dim: int = 10
    num_additions: int = 2
    seed: int = 12334
    addition_method: AdditionMethod = AdditionMethod.DIRECT
    max_target_bond_dim: int = 20
    min_target_bond_dim: int = 10
    step_target_bond_dim: int = 5
    low: float = -0.5
    high: float = 1.0

def run_additions(params: AdditionParams
                  ) -> Results:
    """
    Runs the TTNS addition simulation.

    Args:
        params (AdditionParams): The parameters for the simulation.

    Returns:
        Results: The results of the simulation.
    """
    results = Results()
    res_dtypes = (int, float, float)
    bd_range = range(params.min_target_bond_dim,
                     params.max_target_bond_dim + 1,
                     params.step_target_bond_dim)
    results.initialize(dict(zip(RES_IDS, res_dtypes)),
                       len(bd_range) - 1,
                       with_time=False)
    ttns = random_ttns(params.structure,
                       params.sys_size,
                       params.phys_dim,
                       params.bond_dim,
                       seed=params.seed,
                       distribution=RandomDistribution.UNIFORM,
                       low=params.low,
                       high=params.high)
    ref_ttns = ttns.scale(params.num_additions,
                          inplace=False)
    for index, bd in enumerate(bd_range):
        temp_ttns = deepcopy(ttns)
        ttns_list = [temp_ttns for _ in range(params.num_additions)]
        args = []
        if params.addition_method == AdditionMethod.DENSITY_MATRIX:
            args.append(SVDParameters(max_bond_dim=bd))
        elif params.addition_method == AdditionMethod.DIRECT_TRUNCATE:
            args.append(TruncationMethod.RECURSIVE)
            args.append(SVDParameters(max_bond_dim=bd))

        start = time()
        added_ttn = add_ttns(ttns_list,
                             params.addition_method,
                             *args)
        run_time = time() - start
        added_ttn = TreeTensorNetworkState.from_ttn(added_ttn)
        error = added_ttn.distance(ref_ttns,
                                   normalise=True)
        res_values = (bd, error, run_time)
        for res_id, res_value in zip(RES_IDS, res_values):
            results.set_element(res_id, index, res_value)
    return results

def run_and_save(params: AdditionParams, save_path: str) -> None:
    """
    Runs the TTNS addition simulation and saves the results.

    Args:
        params (AdditionParams): The parameters for the simulation.
        save_path (str): The path to save the results to.
    """
    results = run_additions(params)
    param_hash = params.get_hash()
    filepath = os.path.join(save_path, f"{param_hash}.h5")
    with File(filepath, "w") as file:
        results.save_to_h5(file)
        params.save_to_h5(file)

if __name__ == "__main__":
    script_main(run_and_save,
                parameter_class=AdditionParams)
