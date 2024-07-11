
from typing import Tuple
from argparse import ArgumentParser
import re

import numpy as np
import h5py
import matplotlib.pyplot as plt

from user_guide_tdvp_util import create_file_path

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

def choose_colour(max_bd: int) -> Tuple[float, float, float]:
    if max_bd == 1:
        return colours()["mblue"]
    elif max_bd == 2:
        return colours()["morange"]
    elif max_bd == 3:
        return colours()["mgreen"]
    elif max_bd == 4:
        return colours()["mred"]

def get_filepaths() -> Tuple[str, str, str, str]:
    parser = ArgumentParser()
    parser.add_argument("filepath", nargs=1, type=str)
    args = vars(parser.parse_args())
    filepath = args["filepath"][0]
    load_path = create_file_path(filepath)
    save_path_error = create_file_path(filepath, "_error_time_tdvp.pdf")
    save_path_bonddim = create_file_path(filepath, "_bond_dim_time_tdvp.pdf")
    return load_path, save_path_error, save_path_bonddim

def plot_error(file: h5py.File, savepath: str):
    times = file["times"][:]
    exact_results = file["exact"]["magn"][:]
    figure = plt.figure(figsize=FIGSIZE)
    fo1tdvp_group = file["fo1tdvp"]
    for max_bd in fo1tdvp_group:
        max_bd_value = fo1tdvp_group[max_bd].attrs["max_bd"]
        results = fo1tdvp_group[max_bd]["magn"][:]
        error = np.abs(results - exact_results)
        plt.semilogy(times, error, color=choose_colour(max_bd_value),
                     linestyle=":")
        plt.plot([],[],color=choose_colour(max_bd_value),
                 label=f"Max BD: {max_bd_value}")
    for max_bd in file["so1tdvp"]:
        max_bd_value = file["so1tdvp"][max_bd].attrs["max_bd"]
        results = file["so1tdvp"][max_bd]["magn"][:]
        error = np.abs(results - exact_results)
        plt.semilogy(times, error, color=choose_colour(max_bd_value),
                     linestyle="--")
    for max_bd in file["so2tdvp"]:
        max_bd_value = file["so2tdvp"][max_bd].attrs["max_bd"]
        results = file["so2tdvp"][max_bd]["magn"][:]
        error = np.abs(results - exact_results)
        plt.semilogy(times, error, color=choose_colour(max_bd_value),
                     linestyle="-")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(savepath, format="pdf", bbox_inches='tight')

def plot_bonddim(file: h5py.File, savepath: str):
    times = file["times"][:]
    figure = plt.figure(figsize=FIGSIZE)
    fo1tdvp_group = file["fo1tdvp"]
    for max_bd in fo1tdvp_group:
        max_bd_value = fo1tdvp_group[max_bd].attrs["max_bd"]
        results = fo1tdvp_group[max_bd]["bond_dim"][r"('0', '00')"][:]
        plt.plot(times, results, color=choose_colour(max_bd_value),
                 linestyle=":")
        plt.plot([],[],color=choose_colour(max_bd_value),
                 label=f"Max BD: {max_bd_value}")
    for max_bd in file["so1tdvp"]:
        max_bd_value = file["so1tdvp"][max_bd].attrs["max_bd"]
        results = file["so1tdvp"][max_bd]["bond_dim"][r"('0', '00')"][:]
        plt.plot(times, results, color=choose_colour(max_bd_value),
                 linestyle="--")
    for max_bd in file["so2tdvp"]:
        max_bd_value = file["so2tdvp"][max_bd].attrs["max_bd"]
        results = file["so2tdvp"][max_bd]["bond_dim"][r"('0', '00')"][:]
        plt.plot(times, results, color=choose_colour(max_bd_value),
                 linestyle="-")
    plt.xlabel("Time")
    plt.ylabel("Bond Dimension")
    plt.legend()
    plt.savefig(savepath, format="pdf", bbox_inches='tight')

def plot():
    load_path, save_path_error, save_path_bonddim = get_filepaths()
    with h5py.File(load_path, "r") as file:
        plot_error(file, save_path_error)
        plot_bonddim(file, save_path_bonddim)

if __name__ == "__main__":
    plot()
        