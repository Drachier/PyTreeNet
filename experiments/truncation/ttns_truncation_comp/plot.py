"""
Plots the results from the truncation comparison experiments.
"""
import sys
import os

from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt

from pytreenet.util.plotting.plotables.standard_plottable import (StandardPlottable,
                                                                  combine_equivalent_standard_plottables)
from pytreenet.util.plotting.plotables.multiplot import ConvergingPlottable
from pytreenet.util.experiment_util.metadata_file import MetadataFilter
from pytreenet.core.truncation import TruncationMethod
from pytreenet.time_evolution.results import Results
from pytreenet.util.plotting.line_config import LineConfig
from pytreenet.special_ttn.special_states import TTNStructure

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

def method_colour(params: TruncationParams) -> str:
    """
    Assigns a color to the truncation method for plotting purposes.

    Args:
        params (TruncationParams): The parameters of the simulation.

    Returns:
        str: A color code for the truncation method.
    """
    if params.trunc_method is TruncationMethod.VARIATIONAL:
        return "tab:green"
    if params.random_trunc:
        if params.trunc_method is TruncationMethod.SVD:
            return "tab:orange"
        if params.trunc_method is TruncationMethod.RECURSIVE:
            return "tab:red"
    if params.trunc_method is TruncationMethod.SVD:
        return "tab:blue"
    if params.trunc_method is TruncationMethod.RECURSIVE:
        return "tab:purple"
    raise ValueError(f"Unknown truncation method: {params.trunc_method}")

def build_lineconfig(params: TruncationParams) -> LineConfig:
    """
    Builds a LineConfig object for plotting based on the simulation parameters.

    Args:
        params (TruncationParams): The parameters of the simulation.
    
    Returns:
        LineConfig: The configuration for plotting lines.
    """
    lc = LineConfig()
    lc.color = method_colour(params)
    lc.label = method_name(params)
    lc.marker = "."
    return lc

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
    out: dict[str,
              tuple[list[StandardPlottable],
                    list[StandardPlottable],
                    list[StandardPlottable]]] = {}
    for params, results in params_results:
        lc = build_lineconfig(params)
        bd_vs_err = StandardPlottable.from_simulation_result(results, params,
                                                             extract_bd_vs_err)
        bd_vs_runtime = StandardPlottable.from_simulation_result(results, params,
                                                                 extract_bd_vs_runtime)
        bd_vs_err.line_config = lc
        bd_vs_runtime.line_config = lc
        m_name = method_name(params)
        if m_name not in out:
            out[m_name] = ([], [])
        out[m_name][0].append(bd_vs_err)
        out[m_name][1].append(bd_vs_runtime)
    actual_out: dict[str, tuple[StandardPlottable, StandardPlottable, StandardPlottable]] = {}
    for method, value in out.items():
        conv_plots = (ConvergingPlottable.from_multiple_standards(value[0]),
                       ConvergingPlottable.from_multiple_standards(value[1]))
        stand_plts = (conv_plots[0].average_results(),
                      conv_plots[1].average_results())
        comb_plot = combine_equivalent_standard_plottables(stand_plts[1],stand_plts[0])
        actual_out[method] = (stand_plts[0], stand_plts[1], comb_plot)
    for label, pltbs in actual_out.items():
        for pl in pltbs:
            pl.line_config.label = label
    return actual_out

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
    structures = (TTNStructure.MPS, TTNStructure.BINARY,
                  TTNStructure.FTPS, TTNStructure.TSTAR)
    sys_sizes = (50,7,8,20)
    for structure, sys_size in zip(structures, sys_sizes):
        md_filter = MetadataFilter()
        md_filter.add_to_criterium("structure", structure.value)
        md_filter.add_to_criterium("sys_size", sys_size)
        md_filter.add_to_criterium("bond_dim", 80)
        save_path = os.path.join(data_dir, "plots")
        save_path = os.path.join(save_path, f"truncation_comparison_{structure.value}.pdf")
        plot_all(md_filter, data_dir, save_path=save_path)
