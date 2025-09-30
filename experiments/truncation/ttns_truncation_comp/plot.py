"""
Plots the results from the truncation comparison experiments.
"""
import sys
import os

from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt

from pytreenet.util.plotting.plotables.standard_plottable import StandardPlottable
from pytreenet.util.experiment_util.metadata_file import MetadataFilter
from pytreenet.core.truncation import TruncationMethod
from pytreenet.time_evolution.results import Results

from sim_script import (TruncationParams,
                        RES_IDS)

def method_name(params: TruncationParams) -> str:
    """
    Generates a descriptive name for the truncation method used in the simulation.

    Args:
        params (TruncationParams): The parameters of the simulation.

    Returns:
        str: A descriptive name for the truncation method.
    """
    name = params.trunc_method.value.capitalize()
    if params.random_trunc:
        name = "Random " + name
    return name

def extract_bd_vs_err(result: Results
                      ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Extracts the bond dimension vs. error data from the simulation results.

    Args:
        result (Results): The simulation results containing error data.

    Returns:
        tuple[NDArray[np.floating], NDArray[np.floating]]: Two arrays containing
        bond dimensions and corresponding errors.
    """
    bond_dims = result.operator_result(RES_IDS[0])
    errors = result.operator_result(RES_IDS[1])
    return bond_dims, errors

def extract_bd_vs_runtime(result: Results
                         ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Extracts the bond dimension vs. runtime data from the simulation results.

    Args:
        result (Results): The simulation results containing runtime data.
    
    Returns:
        tuple[NDArray[np.floating], NDArray[np.floating]]: Two arrays containing
        bond dimensions and corresponding runtimes.
    """
    bond_dims = result.operator_result(RES_IDS[0])
    runtimes = result.operator_result(RES_IDS[2])
    return bond_dims, runtimes

def extract_runtime_vs_err(result: Results
                          ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Extracts the time vs. error data from the simulation results.

    Args:
        result (Results): The simulation results containing error data.

    Returns:
        tuple[NDArray[np.floating], NDArray[np.floating]]: Two arrays containing
        times and corresponding errors.
    """
    times = np.cumsum(result.operator_result(RES_IDS[2]))
    errors = result.operator_result(RES_IDS[1])
    return times, errors

def load_data(md_filter: MetadataFilter,
              directory_path: str
              ) -> dict[str, tuple[StandardPlottable, StandardPlottable, StandardPlottable]]:
    """
    Load data from the specified directory and filter it based on the provided
    metadata filter.

    Args:
        md_filter (MetadataFilter): The metadata filter to apply.
        directory_path (str): The path to the directory containing the data.
    
    Returns:
        dict[str, tuple[StandardPlottable, StandardPlottable, StandardPlottable]]:
            A dictionary mapping the truncation method names to tuples of plottables
            for bond dimension vs. error, bond dimension vs. runtime, and time vs. error.
    """
    params_results = md_filter.load_valid_results_and_parameters(directory_path,
                                                                 parameter_class=TruncationParams)
    out = {}
    for params, results in params_results:
        bd_vs_err = StandardPlottable.from_simulation_result(results, params,
                                                             extract_bd_vs_err)
        bd_vs_runtime = StandardPlottable.from_simulation_result(results, params,
                                                                 extract_bd_vs_runtime)
        runtime_vs_err = StandardPlottable.from_simulation_result(results, params,
                                                                  extract_runtime_vs_err)
        m_name = method_name(params)
        out[m_name] = (bd_vs_err, bd_vs_runtime, runtime_vs_err)
        for pltb in out[m_name]:
            pltb.line_config.label = m_name
    return out

def plot_all(md_filter: MetadataFilter,
             directory_path: str,
             save_path: str | None = None
             ) -> None:
    """
    Plots all the data from the specified directory after filtering it based on
    the provided metadata filter.

    Args:
        md_filter (MetadataFilter): The metadata filter to apply.
        directory_path (str): The path to the directory containing the data.
        save_path (str | None, optional): The path to save the plots. If None,
            the plots will be displayed instead of being saved. Defaults to None.
    """
    data = load_data(md_filter, directory_path)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    x_labels = ("Bond Dimension", "Bond Dimension", "Runtime [s]")
    y_labels = ("Error", "Runtime [s]", "Error")
    xlog = (False, False, True)
    ylog = (True, True, True)
    for method, (bd_vs_err, bd_vs_runtime, runtime_vs_err) in data.items():
        bd_vs_err.plot_on_axis(axs[0])
        bd_vs_runtime.plot_on_axis(axs[1])
        runtime_vs_err.plot_on_axis(axs[2])
    for ax, x_label, y_label, x_log, y_log in zip(axs, x_labels, y_labels, xlog, ylog):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")
    axs[0].legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <data_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    md_filter = MetadataFilter()
    md_filter.add_to_criterium("structure", "mps")
    md_filter.add_to_criterium("sys_size", 20)
    md_filter.add_to_criterium("random_trunc", [True, False])
    save_path = os.path.join(data_dir, "plots")
    save_path = os.path.join(save_path, "truncation_comparison.pdf")
    plot_all(md_filter, data_dir, save_path=save_path)
