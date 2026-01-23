"""
This module contains the plotting for the basic comparison experiments.
"""
from argparse import ArgumentParser
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

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
    fig_avg, axes_avg = plt.subplots(2, 3,
                                     figsize=size,
                                     sharex=True,
                                     sharey="row",
                                     height_ratios=[5, 1])
    fig_num, axes_num = plt.subplots(2, 3,
                                     figsize=size, sharex=True,
                                     height_ratios=[5, 1])
    modes = (RandomGenerationMode.EQUAL,
             RandomGenerationMode.SOME_SHARED,
             RandomGenerationMode.UNIQUE)
    finders = (TTNOFinder.SGE,
               TTNOFinder.SGE_PURE,
               TTNOFinder.BIPARTITE)
    for i, mode in enumerate(modes):
        ax_avg_up = axes_avg[0, i]
        ax_avg_down = axes_avg[1, i]
        ax_num_up = axes_num[0, i]
        ax_num_down = axes_num[1, i]
        for finder in finders:
            filename = create_filename(filepath, mode, finder)
            n_terms, diff_avgs, diff_nums = load_data(filename)
            kwargs = plot_kwargs(finder)
            ax_avg_up.plot(n_terms, np.asarray(diff_avgs), ls="", **kwargs)
            ax_avg_down.plot(n_terms, np.asarray(diff_avgs), ls="", **kwargs)
            ax_num_up.plot(n_terms, np.asarray(diff_nums), ls="", **kwargs)
            ax_num_down.plot(n_terms, np.asarray(diff_nums), ls="", **kwargs)
        # Split axes for avg diff
        ax_avg_up.set_ylim(1e-6+1e-7, 1e-2)
        ax_avg_down.set_ylim(0, 1e-10)
        ax_avg_up.spines.bottom.set_visible(False)
        ax_avg_down.spines.top.set_visible(False)
        ax_avg_up.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax_avg_down.xaxis.tick_bottom()
        ax_avg_down.set_yticks([0])
        d = 0.5
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_avg_up.plot([0, 1], [0, 0], transform=ax_avg_up.transAxes, **kwargs)
        ax_avg_down.plot([0, 1], [1, 1], transform=ax_avg_down.transAxes, **kwargs)
        # Split axes for num diff
        ax_num_up.set_ylim(1e-1+1e-2, 1e3)
        ax_num_down.set_ylim(0, 1e-2)
        ax_num_up.spines.bottom.set_visible(False)
        ax_num_down.spines.top.set_visible(False)
        ax_num_up.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax_num_up.tick_params(labeltop=False)
        ax_num_down.xaxis.tick_bottom()
        ax_num_down.set_yticks([0])
        d = 0.5
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_num_up.plot([0, 1], [0, 0], transform=ax_num_up.transAxes, **kwargs)
        ax_num_down.plot([0, 1], [1, 1], transform=ax_num_down.transAxes, **kwargs)
        ax_avg_up.set_yscale("log")
        ax_num_up.set_yscale("log")
        ax_avg_down.set_xlabel("Number of Terms")
        ax_num_down.set_xlabel("Number of Terms")
        if i == 0:
            ax_avg_up.set_ylabel("avg bd")
            ax_num_up.set_ylabel("num diff")
        if i == 2:
            ax_avg_up.legend()
            ax_num_up.legend()
    fig_avg.tight_layout()
    fig_num.tight_layout()
    fig_avg.subplots_adjust(hspace=0.1)
    fig_num.subplots_adjust(hspace=0.1)
    fig_avg.savefig(os.path.join(filepath, "basic_comparison_avg.pdf"))
    fig_num.savefig(os.path.join(filepath, "basic_comparison_num.pdf"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str, nargs=1)
    filepath = vars(parser.parse_args())["filepath"][0]
    plot_basic_comparison(filepath)
