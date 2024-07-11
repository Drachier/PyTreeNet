"""
This script can only be run, once the 2TDVP simulations are done,
in order to obtain the bond dimensions for the 1TDVP.
"""
from argparse import ArgumentParser
from typing import List, Tuple
from os import listdir, path
import time

import h5py
from tqdm import tqdm

from pytreenet.time_evolution.tdvp import (FirstOrderOneSiteTDVP,
                                           SecondOrderOneSiteTDVP,
                                           TDVPAlgorithm)

from user_guide_tdvp_util import (generate_fo_1tdvp,
                                  generate_so_1tdvp)
from user_guide_tdvp_dep_length import (save_times,
                                        save_time_evo_paramters)

def input_handling() -> str:
    """
    Handels the command line argument.
    """
    parser = ArgumentParser()
    parser.add_argument("filepath", nargs=1, type=str)
    args = vars(parser.parse_args())
    filepath = args["filepath"][0]
    return filepath

def generate_2tdvp_filenames(filepath: str) -> List[str]:
    """
    Generate a list of filenames in which the results of the 2TDVP simulations
    are stored.
    """
    tdvp2_files = "so2tdvp"
    directory = path.join(filepath, tdvp2_files)
    return  [path.join(directory, filename)
             for filename in listdir(directory)]

def load_max_bd(file: h5py.File) -> int:
    """
    Load the maximum bond dimension from the file.
    """
    final_bds = [bond_dim[-1] for bond_dim in file["bond_dim"].values()]
    return max(final_bds)

def generate_fo1tdvp_opt_bd(filename: str) -> Tuple[FirstOrderOneSiteTDVP,float,int]:
    """
    Generate a fo1tdvp with the optimal bond dimension as given by the 2TDVP.
    """
    with h5py.File(filename, "r") as file:
        length = file.attrs["length"]
        g = file.attrs["g"]
        max_bd = load_max_bd(file)
        delta_t = file.attrs["delta_t"]
        final_time = file.attrs["final_time"]
        tdvp = generate_fo_1tdvp(length,g,max_bd,delta_t,final_time)
    return tdvp, g, length

def generate_so1tdvp_opt_bd(filename: str) -> Tuple[SecondOrderOneSiteTDVP,float,int]:
    """
    Generate a fo1tdvp with the optimal bond dimension as given by the 2TDVP.
    """
    with h5py.File(filename, "r") as file:
        length = file.attrs["length"]
        g = file.attrs["g"]
        max_bd = load_max_bd(file)
        delta_t = file.attrs["delta_t"]
        final_time = file.attrs["final_time"]
        tdvp = generate_so_1tdvp(length,g,max_bd,delta_t,final_time)
    return tdvp, g, length

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
    filepath = filepath + f"/{orderstr}_opt_bd/length_{length}.hdf5"
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

def main():
    filepath = input_handling()
    filenames = generate_2tdvp_filenames(filepath)
    for filename in tqdm(filenames):
        fotdvp, g, length = generate_fo1tdvp_opt_bd(filename)
        t1 = time.time()
        fotdvp.run(pgbar=False)
        t2 = time.time()
        save_1tdvp_results(filepath, length, t2 - t1,
                           fotdvp, g, 1)
        sotdvp, g, length = generate_so1tdvp_opt_bd(filename)
        t1 = time.time()
        sotdvp.run(pgbar=False)
        t2 = time.time()
        save_1tdvp_results(filepath, length, t2 - t1,
                           sotdvp, g, 2)

if __name__ == "__main__":
    main()