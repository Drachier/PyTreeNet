
from typing import Tuple
import re

import numpy as np
import h5py
import matplotlib.pyplot as plt

from tebd_user_guide_util import input_handling_filepath, create_file_path

FIGSIZE = (4,3.5)

def colours():
    return {"mblue": (0.368417, 0.506779, 0.709798),
            "morange": (0.880722, 0.611041, 0.142051),
            "mgreen": (0.560181, 0.691569, 0.194885),
            "mred": (0.922526, 0.385626, 0.209179),
            "mpurple": (0.528488, 0.470624, 0.701351),
            "mbrown": (0.772079, 0.431554, 0.102387),
            "mpink": (0.910569, 0.506117, 0.755282),
            "mlightblue": (0.676383, 0.866289, 0.803225)}

def get_filepaths() -> Tuple[str, str, str, str]:
    parser = input_handling_filepath()
    args = vars(parser.parse_args())
    filepath = args["filepath"][0]
    load_path = create_file_path(filepath)
    if re.match(r".*\.hdf5", filepath):
        filepath = filepath[:-5]
    save_path_magn = create_file_path(filepath, "_magn_time.pdf")
    save_path_error = create_file_path(filepath, "_error_time.pdf")
    save_path_bonddim = create_file_path(filepath, "_bond_dim_time.pdf")
    return load_path, save_path_magn, save_path_error, save_path_bonddim

def plot_magn(file: h5py.File, savepath: str):
    times = file["times"][:]
    exact_results = file["exact"]["magn"][:]
    figure = plt.figure(figsize=FIGSIZE)
    for max_bd in file:
        if re.match(r"max_bd_\d+", max_bd):
            results = file[max_bd]["magn"][:]
            max_bd_value = file[max_bd].attrs["max_bd"]
            plt.plot(times, results, label=f"Max BD: {max_bd_value}",
                        color = list(colours().values())[max_bd_value])
    plt.xlabel("Time")
    plt.ylabel(r"$\langle M \rangle$")
    plt.legend()
    plt.plot(times, exact_results, label="Exact", c="black")
    plt.savefig(savepath, format="pdf", bbox_inches='tight')

def plot_error(file: h5py.File, savepath: str):
    times = file["times"][:]
    exact_results = file["exact"]["magn"][:]
    figure = plt.figure(figsize=FIGSIZE)
    for max_bd in file:
        if re.match(r"max_bd_\d+", max_bd):
            results = file[max_bd]["magn"][:]
            max_bd_value = file[max_bd].attrs["max_bd"]
            error = np.abs(results - exact_results)
            plt.semilogy(times, error, label=f"Max BD: {max_bd_value}",
                         color = list(colours().values())[max_bd_value])
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(savepath, format="pdf", bbox_inches='tight')

def plot_bonddim(file: h5py.File, savepath: str):
    times = file["times"][:]
    figure = plt.figure(figsize=FIGSIZE)
    for max_bd in file:
        if re.match(r"max_bd_\d+", max_bd):
            results = file[max_bd]["bond_dim"]
            max_bd_value = file[max_bd].attrs["max_bd"]
            for key, value in results.items():
                if key[1:4] == r"'0'" and key[6:10] == r"'00'":
                    bond_dims = value[:]
                    plt.plot(times, bond_dims, label=f"Max BD: {max_bd_value}",
                            color = list(colours().values())[max_bd_value])
    plt.xlabel("Time")
    plt.ylabel("Bond Dimension (0,00)")
    plt.legend()
    plt.savefig(savepath, format="pdf", bbox_inches='tight')

def plot():
    load_path, save_path_magn, save_path_error, save_path_bonddim = get_filepaths()
    with h5py.File(load_path, "r") as file:
        plot_magn(file, save_path_magn)
        plot_error(file, save_path_error)
        plot_bonddim(file, save_path_bonddim)

if __name__ == "__main__":
    plot()
