from argparse import ArgumentParser
from typing import Dict, Tuple
from time import time
from enum import Enum

import h5py
from numpy import ceil
from tqdm import tqdm

from pytreenet.time_evolution.time_evolution import TimeEvolution
from pytreenet.time_evolution.tdvp import TDVPAlgorithm

from user_guide_tdvp_util import (generate_fo_1tdvp,
                                  generate_so_1tdvp,
                                  generate_so_2tdvp)

def save_times(file: h5py.File, evo: TimeEvolution):
    """
    Save the times.
    """
    file.create_dataset("times", data=evo.times())

def save_time_evo_paramters(file: h5py.File, delta_t: float, final_time: float,
                            g: float):
    """
    Save the time evolution parameters.
    """
    file.attrs["delta_t"] = delta_t
    file.attrs["final_time"] = final_time
    file.attrs["g"] = g

def save_so2tdvp_results(filepath: str, length: int, t: float,
                         tdvp: TDVPAlgorithm, g: float):
    """
    Saves the results of the second order two-site TDVP.
    """
    filepath = filepath + f"/so2tdvp/length_{length}.hdf5"
    with h5py.File(filepath, "w") as file:
        save_time_evo_paramters(file, tdvp.time_step_size,
                                tdvp.final_time, g)
        svd_params = tdvp.svd_parameters
        file.attrs["max_bond_dim"] = svd_params.max_bond_dim
        file.attrs["rel_tol"] = svd_params.rel_tol
        file.attrs["total_tol"] = svd_params.total_tol
        file.attrs["length"] = length
        file.attrs["time"] = t
        bd_group = file.create_group("bond_dim")
        for key, value in tdvp.operator_result("bond_dim").items():
            bd_group.create_dataset(str(key), data=value)
        file.create_dataset("magn", data=tdvp.operator_result("magn",
                                                            realise=True))
        save_times(file, tdvp)

def save_1tdvp_results(filepath: str, length: int, t: float,
                       tdvp: TDVPAlgorithm, g: float, order: int):
    """
    Saves the results of the first order TDVP.
    """
    if order == 1:
        orderstr = "fo1tdvp"
    elif order == 2:
        orderstr = "so1tdvp"
    else:
        raise ValueError("Only first and second order 1TDVP are supported!")
    filepath = filepath + f"/{orderstr}/length_{length}.hdf5"
    with h5py.File(filepath, "w") as file:
        save_time_evo_paramters(file, tdvp.time_step_size,
                                tdvp.final_time, g)
        file.attrs["length"] = length
        file.attrs["time"] = t
        bd_group = file.create_group("bond_dim")
        for key, value in tdvp.operator_result("bond_dim").items():
            bd_group.create_dataset(str(key), data=value)
        file.create_dataset("magn", data=tdvp.operator_result("magn",
                                                            realise=True))
        save_times(file, tdvp)

def input_handling():
    """
    Handle command line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("filepath", nargs=1, type=str)
    parser.add_argument("min_length", nargs=1, type=int)
    parser.add_argument("max_length", nargs=1, type=int)
    parser.add_argument("mode", nargs=1, type=int)
    args = vars(parser.parse_args())
    filepath = args["filepath"][0]
    min_length = args["min_length"][0]
    max_length = args["max_length"][0]
    mode = args["mode"][0]
    if mode == 0:
        mode = TDVPMode.FO1
    elif mode == 1:
        mode = TDVPMode.SO1
    elif mode == 2:
        mode = TDVPMode.SO2
    return filepath, min_length, max_length, mode

def parameters() -> Dict:
    return {"delta_t": 0.01, "final_time": 1, "g": 0.1,
            "rel_tol": 1e-10, "total_tol": 1e-10}

class TDVPMode(Enum):
    FO1 = 0
    SO1 = 1
    SO2 = 2
    FO1c = 3
    SO1c = 4

    def to_str(self) -> str:
        if self == TDVPMode.FO1:
            return "fo1tdvp"
        elif self == TDVPMode.SO1:
            return "so1tdvp"
        elif self == TDVPMode.SO2:
            return "so2tdvp"
        elif self == TDVPMode.FO1c:
            return "fo1tdvp_opt_bd"
        elif self == TDVPMode.SO1c:
            return "so1tdvp_opt_bd"
        else:
            raise ValueError("Invalid mode!")
        
    def colour(self) -> Tuple[float, float, float]:
        if self in [TDVPMode.FO1, TDVPMode.SO1, TDVPMode.SO2]:
            return (0.368417, 0.506779, 0.709798) # mblue
        elif self in [TDVPMode.FO1c, TDVPMode.SO1c]:
            return (0.880722, 0.611041, 0.142051) # morange
        else:
            raise ValueError("Invalid mode!")

    def linestyle(self) -> str:
        if self == TDVPMode.FO1 or self == TDVPMode.FO1c:
            return ":"
        elif self == TDVPMode.SO1 or self == TDVPMode.SO1c:
            return "--"
        elif self == TDVPMode.SO2:
            return "-"
        else:
            raise ValueError("Invalid mode!")

def create_tdvp(length: int, mode: TDVPMode) -> TDVPAlgorithm:
    params = parameters()
    delta_t = params["delta_t"]
    final_time = params["final_time"]
    g = params["g"]
    rel_tol = params["rel_tol"]
    total_tol = params["total_tol"]
    max_bond_dim = int(ceil(length / 2))
    if mode == TDVPMode.FO1:
        tdvp = generate_fo_1tdvp(length, g, max_bond_dim, delta_t, final_time)
    elif mode == TDVPMode.SO1:
        tdvp = generate_so_1tdvp(length, g, max_bond_dim, delta_t, final_time)
    elif mode == TDVPMode.SO2:
        tdvp = generate_so_2tdvp(length, g, delta_t, final_time,
                                 max_bond_dim, rel_tol, total_tol)
    else:
        raise ValueError("Invalid mode!")
    return tdvp

def save_results(filepath: str, length: int, t: float,
                 tdvp: TDVPAlgorithm, g: float, mode: TDVPMode):
    if mode == TDVPMode.FO1:
        save_1tdvp_results(filepath, length, t, tdvp, g, 1)
    elif mode == TDVPMode.SO1:
        save_1tdvp_results(filepath, length, t, tdvp, g, 2)
    elif mode == TDVPMode.SO2:
        save_so2tdvp_results(filepath, length, t, tdvp, g)

def run_sim(filepath, min_length, max_length, mode: TDVPMode):
    g = parameters()["g"]
    for length in tqdm(range(min_length, max_length + 1)):
        tdvp = create_tdvp(length, mode)
        t0 = time()
        tdvp.run(pgbar=False)
        t1 = time()
        save_results(filepath, length, t1-t0, tdvp, g, mode)

def main():
    filepath, min_length, max_length, mode = input_handling()
    run_sim(filepath, min_length, max_length, mode)

if __name__ == "__main__":
    main()
