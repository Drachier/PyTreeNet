
from typing import Tuple, List
from argparse import ArgumentParser

import h5py
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors, cm

from user_guide_tdvp_util import create_file_path
from user_guide_tdvp_dep_length import TDVPMode

FIGSIZE = (4,3.5)

def input_handling() -> Tuple[str, int, int]:
    parser = ArgumentParser()
    parser.add_argument("filepath", nargs=1, type=str)
    parser.add_argument("min_length", nargs=1, type=int)
    parser.add_argument("max_length", nargs=1, type=int)
    args = vars(parser.parse_args())
    filepath = args["filepath"][0]
    min_length = args["min_length"][0]
    max_length = args["max_length"][0]
    return filepath, min_length, max_length

def get_savepaths(filepath: str) -> Tuple[str, str]:
    save_path_time = create_file_path(filepath, "_time.pdf")
    save_path_bonddim = create_file_path(filepath, "_bond_dim_time.pdf")
    return save_path_time, save_path_bonddim

def get_load_path(filepath: str, length: int, mode: TDVPMode) -> str:
    return filepath + f"/{mode.to_str()}/length_{length}.hdf5"

def plot_bonddim_one_tdvp(ax: plt.Axes, filepath: str, length: int,
                          mode: TDVPMode, max_length: int):
    try:
        with h5py.File(get_load_path(filepath, length, mode), "r") as file:
            assert length ==  file.attrs["length"]
            times = file["times"][:]
            bds = file["bond_dim"][r"('0', '00')"][:]
            colour = colormaps.get_cmap("inferno")(length / max_length)
            ax.plot(times, bds, linestyle=mode.linestyle(), color=colour)
    except FileNotFoundError:
        pass

def plot_bonddim(filepath: str, savepath: str,
                 min_length: int, max_length: int):
    max_length = 14
    figure, ax = plt.subplots(figsize=FIGSIZE)
    for length in range(min_length, max_length + 1):
        for mode in [TDVPMode.FO1, TDVPMode.SO1, TDVPMode.SO2]:
            plot_bonddim_one_tdvp(ax, filepath, length, mode, max_length)
    plt.xlabel("Time")
    plt.ylabel("Bond Dimension")
    plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap='inferno'), ax=ax,
                label=r"$L / L_{\max}$")
    plt.savefig(savepath, format="pdf", bbox_inches='tight')

def load_times_wrt_length(filepath: str, mode: TDVPMode,
                          min_length: int, max_length: int) -> Tuple[List[int], List[float]]:
    times = []
    lengths = []
    for length in range(min_length, max_length + 1):
        try:
            with h5py.File(get_load_path(filepath, length, mode), "r") as file:
                times.append(file.attrs["time"])
                lengths.append(length)
        except FileNotFoundError:
            print(f"File for length {length} not found.")
            pass
    return lengths, times

def plot_time(filepath: str, savepath: str,
              min_length: int, max_length: int):
    figure = plt.figure(figsize=FIGSIZE)
    for mode in TDVPMode:
        lengths, times = load_times_wrt_length(filepath, mode,
                                               min_length, max_length)
        plt.semilogy(lengths, times, linestyle=mode.linestyle(),
                 color=mode.colour(), label=mode.to_str())
    plt.xlabel("Length $L$")
    plt.ylabel("Runtime")
    plt.legend()
    plt.savefig(savepath, format="pdf", bbox_inches='tight')

def plot():
    filepath, min_length, max_length = input_handling()
    save_path_time, save_path_bonddim = get_savepaths(filepath)
    plot_bonddim(filepath, save_path_bonddim, min_length, max_length)
    plot_time(filepath, save_path_time, min_length, max_length)

if __name__ == "__main__":
    plot()
        