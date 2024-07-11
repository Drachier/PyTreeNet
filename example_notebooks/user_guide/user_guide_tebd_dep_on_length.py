
from typing import Union, Tuple
from argparse import ArgumentParser

import h5py
import numpy as np
from tqdm import tqdm

from tebd_user_guide_util import input_handling_filepath, create_tebd, create_file_path

from pytreenet.time_evolution.tebd import TEBD


def input_handling(parser: Union[ArgumentParser,None] = None) -> Tuple[str, int, int]:
    """
    Handles the input parsed from the command line.
    
    In this case we want a file path, a minimum and a maximum length.
    """
    if parser is None:
        parser = ArgumentParser()
    parser = input_handling_filepath(parser)
    parser.add_argument("min_length", type=int, nargs=1)
    parser.add_argument("max_length", type=int, nargs=1)
    args = vars(parser.parse_args())
    filepath = args["filepath"][0]
    filepath = create_file_path(filepath)
    min_length = args["min_length"][0]
    max_length = args["max_length"][0]
    return filepath, min_length, max_length

def save_metadata(file: h5py.File, min_length: int, max_length: int,
                  g: float, time_step_size: float, final_time: float,
                  rel_tol: float, total_tol: float):
    """
    Save the metadata.
    """
    file.attrs["min_length"] = min_length
    file.attrs["max_length"] = max_length
    file.attrs["g"] = g
    file.attrs["time_step_size"] = time_step_size
    file.attrs["final_time"] = final_time
    file.attrs["rel_tol"] = rel_tol
    file.attrs["total_tol"] = total_tol

def obtain_average_bond_dims(tebd: TEBD):
    """
    Obtain the average bond dimension.
    """
    total_bond_dim = np.zeros_like(tebd.times())
    for bond_dim in tebd.operator_result("bond_dim").values():
        total_bond_dim += np.array(bond_dim)
    num_bonds = len(tebd.state.nodes) - 1
    return total_bond_dim / num_bonds

def save_results(file: h5py.File, length: int, tebd: TEBD):
    """
    Saves the results of this time-evolution.
    """
    group = file.create_group(f"length_{length}")
    group.attrs["length"] = length
    group.create_dataset("bond_dim", data=obtain_average_bond_dims(tebd))
    group.create_dataset("magn", data=tebd.operator_result("magn",
                                                           realise=True))

def save_times(file: h5py.File, tebd: TEBD):
    """
    Save the times.
    """
    file.create_dataset("times", data=tebd.times())

def run(filepath: str, min_length: int, max_length: int):
    """
    Run the experiment.
    """
    with h5py.File(filepath, "w") as file:
        g = 0.1
        time_step_size = 0.001
        final_time = 1
        rel_tol = 1e-10
        total_tol = 1e-10
        save_metadata(file, min_length, max_length,
                      g, time_step_size, final_time,
                      rel_tol, total_tol)
        for length in tqdm(range(min_length, max_length + 1)):
            max_bond_dim = length
            tebd = create_tebd(length,
                               g,
                               time_step_size,
                               final_time,
                               max_bond_dim,
                               rel_tol,
                               total_tol)
            tebd.run(pgbar=False)
            save_results(file, length, tebd)
        save_times(file, tebd)

def main():
    filepath, min_length, max_length = input_handling()
    run(filepath, min_length, max_length)

if __name__ == "__main__":
    main()