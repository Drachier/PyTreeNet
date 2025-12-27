"""
This module contains the plotting for the basic comparison experiments.
"""
from argparse import ArgumentParser
import os

import h5py
import matplotlib.pyplot as plt

from pytreenet.util.plotting.configuration import (DocumentStyle,
                                                   set_size,
                                                   config_matplotlib_to_latex)

from sim_script import (RandomGenerationMode, TTNOFinder,
                         create_filename)

def plot_kwargs(finder: TTNOFinder) -> dict:
    """
    Get the plotting keyword arguments for the given finder.

    Args:
        finder (TTNOFinder): The TTNOFinder enum value.
    
    Returns:
        dict: A dictionary of plotting keyword arguments.
    """
    out_kwargs = {"markersize": 2, "linewidth": 0.5}
    if finder == TTNOFinder.SGE:
        out_kwargs.update({"color": "blue", "marker": "o", "label": "Combined"})
    elif finder == TTNOFinder.SGE_PURE:
        out_kwargs.update({"color": "orange", "marker": "s", "label": "SGE"})
    elif finder == TTNOFinder.BIPARTITE:
        out_kwargs.update({"color": "green", "marker": "^", "label": "Bipartite"})
    else:
        raise ValueError(f"Invalid finder: {finder}")
    return out_kwargs

def load_data(filepath: str) -> tuple[list, list, list]:
    """
    Load the data from the given filepath for the specified mode and finder.

    Args:
        filepath (str): Path to the HDF5 file containing the data.

    Returns:
        dict: Loaded data corresponding to the specified mode and finder.
    """
    n_terms = []
    diff_avgs = []
    diff_nums = []
    with h5py.File(filepath, "r") as f:
        max_num_terms = int(f.attrs["max_num_terms"])
        for num_terms in range(1, max_num_terms + 1):
            grp = f[f"run_with_{num_terms}_terms"]
            data_svd = grp["svd_bond_dim"][:]
            data_non_svd = grp["state_diag_bond_dim"][:]
            diff = data_non_svd - data_svd
            diff_avg = diff.mean()
            diff_num = (diff > 0).sum()
            n_terms.append(num_terms)
            diff_avgs.append(diff_avg)
            diff_nums.append(diff_num)
    return (n_terms, diff_avgs, diff_nums)

def plot_basic_comparison(filepath: str):
    """
    Plot the basic comparison results from the given filepath.

    Args:
        filepath (str): Path to the directory containing the data files.
    """
    config_matplotlib_to_latex(DocumentStyle.THESIS)
    size = set_size(DocumentStyle.THESIS, subplots=(1, 3))
    size = (size[0], size[1] * 2)
    fig_avg, axes_avg = plt.subplots(1, 3, figsize=size, sharey=True)
    fig_num, axes_num = plt.subplots(1, 3, figsize=size, sharey=True)
    modes = (RandomGenerationMode.EQUAL,
             RandomGenerationMode.SOME_SHARED,
             RandomGenerationMode.UNIQUE)
    finders = (TTNOFinder.SGE,
               TTNOFinder.SGE_PURE,
               TTNOFinder.BIPARTITE)
    for i, mode in enumerate(modes):
        ax_avg = axes_avg[i]
        ax_num = axes_num[i]
        for finder in finders:
            filename = create_filename(filepath, mode, finder)
            n_terms, diff_avgs, diff_nums = load_data(filename)
            kwargs = plot_kwargs(finder)
            ax_avg.plot(n_terms, diff_avgs, **kwargs)
            ax_num.plot(n_terms, diff_nums, **kwargs)
        ax_avg.set_xlabel("Number of Terms")
        ax_num.set_xlabel("Number of Terms")
        if i == 0:
            ax_avg.set_ylabel("avg bd")
            ax_num.set_ylabel("num diff")
        if i == 2:
            ax_avg.legend()
            ax_num.legend()
    fig_avg.tight_layout()
    fig_num.tight_layout()
    fig_avg.savefig(os.path.join(filepath, "basic_comparison_avg.pdf"))
    fig_num.savefig(os.path.join(filepath, "basic_comparison_num.pdf"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str, nargs=1)
    filepath = vars(parser.parse_args())["filepath"][0]
    plot_basic_comparison(filepath)