"""
Implements the common functionality of the plotting scripts for the SDC
application simulations comparing different root nodes and canonicalisation
schemes for MPS.

The distribution-specific part is the `ApplicationParams` subclass, which is
passed to the functions as the `parameter_class` argument.
"""
from __future__ import annotations
from typing import Callable
import sys
import os

from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt

from pytreenet.util.plotting.plotables.standard_plottable import StandardPlottable
from pytreenet.util.plotting.plotables.multiplot import ConvergingPlottable
from pytreenet.util.experiment_util.metadata_file import MetadataFilter
from pytreenet.ttns.ttns_ttno.application import ApplicationMethod
from pytreenet.time_evolution.results import Results
from pytreenet.util.plotting.line_config import LineConfig
from pytreenet.util.plotting.configuration import (DocumentStyle,
                                                   set_size,
                                                   config_matplotlib_to_latex
                                                   )

from application_common import ApplicationParams
from canonicalisation_common import (CanonicalisationMode,
                                     ElementDistribution,
                                     ROOT_CONFIGS,
                                     config_res_ids,
                                     BOND_DIM_ID)

CONFIG_COLOURS = ("tab:blue",
                  "tab:orange",
                  "tab:green",
                  "tab:red",
                  "tab:purple",
                  "tab:brown")

def config_label(config) -> str:
    """
    Generates a descriptive label for a configuration.

    Args:
        config (RootConfig): The configuration to describe.

    Returns:
        str: A descriptive label of the root node and canonicalisation.
    """
    root = "left root" if config.root_is_first_site else "right root"
    if config.canonicalisation is CanonicalisationMode.NONE:
        centre = "no canonicalisation"
    elif config.canonicalisation is CanonicalisationMode.ROOT:
        centre = "centre: root"
    else:
        centre = "centre: " + ("right" if config.root_is_first_site else "left")
    return root + ", " + centre

def build_lineconfig(config) -> LineConfig:
    """
    Builds a LineConfig object for plotting based on the configuration.

    Args:
        config (RootConfig): The configuration of the instance.

    Returns:
        LineConfig: The configuration for plotting lines.
    """
    lc = LineConfig()
    lc.color = CONFIG_COLOURS[ROOT_CONFIGS.index(config)]
    lc.label = config_label(config)
    lc.marker = "."
    return lc

def make_error_extractor(trunc_error_id: str) -> Callable:
    """
    Creates an extraction function for the bond dimension vs. error data of a
    single configuration.

    Args:
        trunc_error_id (str): The identifier of the error result of the
            configuration.

    Returns:
        Callable: The extraction function taking only the simulation results.
    """
    def extract_bd_vs_err(result: Results
                          ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        return (result.operator_result(BOND_DIM_ID),
                result.operator_result(trunc_error_id))
    return extract_bd_vs_err

def make_runtime_extractor(run_time_id: str) -> Callable:
    """
    Creates an extraction function for the bond dimension vs. runtime data of
    a single configuration.

    Args:
        run_time_id (str): The identifier of the runtime result of the
            configuration.

    Returns:
        Callable: The extraction function taking only the simulation results.
    """
    def extract_bd_vs_runtime(result: Results
                              ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        return (result.operator_result(BOND_DIM_ID),
                result.operator_result(run_time_id))
    return extract_bd_vs_runtime

def load_data(md_filter: MetadataFilter,
              directory_path: str,
              parameter_class: type[ApplicationParams]
              ) -> dict[str, tuple[StandardPlottable, StandardPlottable]]:
    """
    Load data from the specified directory and filter it based on the provided
    metadata filter.

    Args:
        md_filter (MetadataFilter): The metadata filter to apply.
        directory_path (str): The path to the directory containing the data.
        parameter_class (type[ApplicationParams]): The class of the simulation
            parameters of the data.

    Returns:
        dict[str, tuple[StandardPlottable, StandardPlottable]]: A dictionary
        mapping the configuration names to tuples of plottables for bond
        dimension vs. error and bond dimension vs. runtime.
    """
    params_results = md_filter.load_valid_results_and_parameters(directory_path,
                                                                 parameter_class=parameter_class)
    config_data = {config.name: ([], []) for config in ROOT_CONFIGS}
    for params, results in params_results:
        for config in ROOT_CONFIGS:
            trunc_error_id, run_time_id = config_res_ids(config)
            lc = build_lineconfig(config)
            bd_vs_err = StandardPlottable.from_simulation_result(results,
                                                                 params,
                                                                 make_error_extractor(trunc_error_id))
            bd_vs_runtime = StandardPlottable.from_simulation_result(results,
                                                                     params,
                                                                     make_runtime_extractor(run_time_id))
            bd_vs_err.line_config = lc
            bd_vs_runtime.line_config = lc
            config_data[config.name][0].append(bd_vs_err)
            config_data[config.name][1].append(bd_vs_runtime)
    out = {}
    for config in ROOT_CONFIGS:
        err_plt = ConvergingPlottable.from_multiple_standards(config_data[config.name][0]).average_results()
        runtime_plt = ConvergingPlottable.from_multiple_standards(config_data[config.name][1]).average_results()
        err_plt.line_config.label = config_label(config)
        runtime_plt.line_config.label = config_label(config)
        out[config.name] = (err_plt, runtime_plt)
    return out

def plot_all(md_filter: MetadataFilter,
             directory_path: str,
             parameter_class: type[ApplicationParams],
             save_path: str | None = None
             ) -> None:
    """
    Plots the bond dimension vs. error and bond dimension vs. runtime data for
    all configurations after filtering the data based on the metadata filter.

    Args:
        md_filter (MetadataFilter): The metadata filter to apply.
        directory_path (str): The path to the directory containing the data.
        parameter_class (type[ApplicationParams]): The class of the simulation
            parameters of the data.
        save_path (str | None, optional): The path to save the plot. If None,
            the plot will be displayed instead of being saved. Defaults to None.
    """
    data = load_data(md_filter, directory_path, parameter_class)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    x_labels = ("Bond Dimension", "Bond Dimension")
    y_labels = ("Error", "Runtime [s]")
    xlog = (False, False)
    ylog = (True, True)
    for _, (bd_vs_err, bd_vs_runtime) in data.items():
        bd_vs_err.plot_on_axis(axs[0])
        bd_vs_runtime.plot_on_axis(axs[1])
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

def standard_plot_main(parameter_class: type[ApplicationParams]) -> None:
    """
    Standard main function for the plotting scripts. Plots the SDC
    canonicalisation comparison for all distributions and starting bond
    dimensions.

    Args:
        parameter_class (type[ApplicationParams]): The class of the simulation
            parameters of the data.
    """
    if len(sys.argv) < 2:
        print("Usage: python plot.py <data_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    bond_dims = (10, 20, 40, 50)
    for distribution in ElementDistribution:
        for bond_dim in bond_dims:
            print(f"Plotting for distribution: {distribution.value},"
                  + f" bond dimension: {bond_dim}")
            md_filter = MetadataFilter()
            md_filter.add_to_criterium("phys_dim", 3)
            md_filter.add_to_criterium("structure", "mps")
            md_filter.add_to_criterium("sys_size", 50)
            md_filter.add_to_criterium("bond_dim", bond_dim)
            md_filter.add_to_criterium("appl_method",
                                       ApplicationMethod.HALF_DENSITY_MATRIX.value)
            md_filter.add_to_criterium("distribution", distribution.value)
            save_path = os.path.join(data_dir, "plots")
            save_path = os.path.join(save_path,
                                     "sdc_canonicalisation_comparison_"
                                     + f"{distribution.value}_bd{bond_dim}.pdf")
            plot_all(md_filter, data_dir, parameter_class, save_path=save_path)
