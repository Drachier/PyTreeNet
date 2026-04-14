"""
Script to investigate the TTNO generated for the circuit simulation.
"""
from __future__ import annotations
import os
import fnmatch
import argparse
from itertools import product

import numpy as np

from pytreenet.ttno.ttno_class import TTNO
from pytreenet.special_ttn.special_states import TTNStructure

from sim_script import level_file_name, ttno_dir

def load_ttnos(dir_path: str) -> list[TTNO]:
    """
    Load the TTNO representing the circuit.

    Args:
        dir_path (str): The directory holding the TTNO save files.

    Returns:
        list[TTNO]: The TTNO loaded from the directory.
    """
    # Half the files are .json and half are .npz
    num_ttno = len(fnmatch.filter(os.listdir(dir_path), "*.json"))
    ttnos = []
    for idx in range(num_ttno):
        filename = os.path.join(dir_path, level_file_name(idx))
        single_ttno = TTNO.load(filename)
        ttnos.append(single_ttno)
    return ttnos

def ttno_info(ttnos: list[TTNO]) -> str:
    """
    Obtain a string showing the desired information about the TTNOs
    """
    out = ""
    avg_bds = []
    max_bds = []
    for ttno in ttnos:
        avg_bd = ttno.avg_bond_dim()
        avg_bds.append(avg_bd)
        max_bd = ttno.max_bond_dim()
        max_bds.append(max_bd)
    #out += "(avg_bd, max_bd):"
    #out += str(list(zip(avg_bds,max_bds)))[1:-1]
    #out += "\n"
    out += "Average of Averages: "
    average_of_averages = np.mean(avg_bds)
    out += f"{average_of_averages}\n"
    out += "Maxmimum of Averages: "
    maximum_average = np.max(avg_bds)
    out += f"{maximum_average}\n"
    out += "Average of Maximums: "
    average_of_maximums = np.mean(max_bds)
    out += f"{average_of_maximums}\n"
    out += "Maximum of Maximums: "
    maximum_of_maximums = np.max(max_bds)
    out += f"{maximum_of_maximums}"
    return out

def get_info_for_one_run(cache_dir_path: str,
                         num_repeats: int,
                         tree_type: TTNStructure,
                         seed: int
                         ) -> str:
    """
    Get the info of the TTNO for one run of the circuit simulation.
    """
    ttno_dir_name = ttno_dir(num_repeats, seed, tree_type)
    dir_path = os.path.join(cache_dir_path, ttno_dir_name)
    ttnos = load_ttnos(dir_path)
    info = ttno_info(ttnos)
    out = f"Tree: {tree_type}, num_repeats: {num_repeats}, seed: {seed}\n"
    out += "-"*20 + "\n"
    out += info
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_directory",
                        type=str,
                        help="The directory in which all the TTNO of all runs are saved.")
    args = parser.parse_args()
    structures = [TTNStructure.MPS, TTNStructure.BINARY, TTNStructure.TSTAR]
    num_repeats = [3]
    seeds = [56, 1234, 1549, 4321, 6846, 27384, 90867]
    for structure, num_repat, seed in product(structures, num_repeats, seeds):
        info = get_info_for_one_run(args.save_directory,
                                    num_repat,
                                    structure,
                                    seed)
        print(info)
        print("*"*20)

if __name__ == "__main__":
    main()
