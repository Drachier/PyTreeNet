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
from pytreenet.util.plotting.line_config import (LineConfig,
                                                 StyleMapping,
                                                 StyleOption)
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.util.plotting.configuration import (DocumentStyle,
                                                   set_size,
                                                   config_matplotlib_to_latex)

from sim_script import (TruncationParams,
                        RES_IDS)

def method_name(params: TruncationParams | tuple[TruncationMethod, bool]) -> str:
    """
    Generates a descriptive name for the truncation method used in the simulation.

    Args:
        params (TruncationParams): The parameters of the simulation.

    Returns:
        str: A descriptive name for the truncation method.
    """
    if isinstance(params, tuple):
        trunc_method = params[0]
        random_trunc = params[1]
    else:
        trunc_method = params.trunc_method
        random_trunc = params.random_trunc
    name = trunc_method.value.capitalize()
    if random_trunc:
        name = "Random " + name
    return name

def method_colour(params: TruncationParams | tuple[TruncationMethod, bool]) -> str:
    """
    Assigns a color to the truncation method for plotting purposes.

    Args:
        params (TruncationParams): The parameters of the simulation.

    Returns:
        str: A color code for the truncation method.
    """
    if isinstance(params, tuple):
        trunc_method = params[0]
        random_trunc = params[1]
    else:
        trunc_method = params.trunc_method
        random_trunc = params.random_trunc

    if trunc_method is TruncationMethod.DENSITYMATRIX:
        return "tab:green"
    if random_trunc:
        if trunc_method is TruncationMethod.SVD:
            return "tab:orange"
        if trunc_method is TruncationMethod.RECURSIVE:
            return "tab:red"
        if trunc_method is TruncationMethod.SVD2SITE:
            return "tab:cyan"
    if trunc_method is TruncationMethod.SVD:
        return "tab:blue"
    if trunc_method is TruncationMethod.RECURSIVE:
        return "tab:purple"
    if trunc_method is TruncationMethod.SVD2SITE:
        return "tab:brown"
    raise ValueError(f"Unknown truncation method: {trunc_method}")

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

def build_style_mapping(md_filter: MetadataFilter) -> StyleMapping:
    """
    Builds a StyleMapping object for plotting based on the metadata filter.

    Args:
        md_filter (MetadataFilter): The metadata filter to use for
            determining styles.

    Returns:
        StyleMapping: The mapping of styles for plotting.
    """
    sm = StyleMapping()
    trunc_methods = md_filter.get_criterium("trunc_method")
    trunc_methods = [TruncationMethod(m) for m in trunc_methods]
    mapping = {(method,False): method_colour((method, False))
               for method in trunc_methods}
    mapping.update({(method,True): method_colour((method, True))
                    for method in trunc_methods
                    if method.randomisable()})
    sm.add_mapping("trunc_method",
                   StyleOption.COLOR,
                   mapping)
    for method in trunc_methods:
        sm.set_label("trunc_method",
                     (method, False),
                     method_name((method, False)))
        if method.randomisable():
            sm.set_label("trunc_method",
                        (method, True),
                        method_name((method, True)))
    return sm

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
                                                                 parameter_class=TruncationParams,
                                                                 allow_non_exist=True)
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
    config_matplotlib_to_latex(style=DocumentStyle.THESIS)
    size = set_size(width=DocumentStyle.THESIS, subplots=(2, 2))

    data = load_data(md_filter, directory_path)
    
    fig, axs = plt.subplots(2, 2, figsize=(size[0], 1.25*size[1]))
    x_labels = ("Bond Dimension", "Bond Dimension", "Runtime [s]")
    y_labels = ("Error", "Runtime [s]", "Error")
    xlog = (False, False, True)
    ylog = (True, True, True)
    for method, (bd_vs_err, bd_vs_runtime, runtime_vs_err) in data.items():
        bd_vs_err.plot_on_axis(axs[0, 0])
        bd_vs_runtime.plot_on_axis(axs[0, 1])
        runtime_vs_err.plot_on_axis(axs[1, 0])
    for ax, x_label, y_label, x_log, y_log in zip(axs.flat[:-1], x_labels, y_labels, xlog, ylog):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")

    # Build and apply custom legend using StyleMapping from MetadataFilter
    style_mapping = build_style_mapping(md_filter)
    axs[1,1].axis('off')  # Hide the last subplot for the legend
    style_mapping.apply_legend(axs[1,1])
    axs[1,1].legend(loc='center left')

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
        md_filter.add_to_criterium("bond_dim", 50)
        md_filter.add_to_criterium("trunc_method", [TruncationMethod.SVD.value,
                                                    TruncationMethod.RECURSIVE.value,
                                                    TruncationMethod.SVD2SITE.value,
                                                    TruncationMethod.DENSITYMATRIX.value])
        save_path = os.path.join(data_dir, "plots")
        save_path = os.path.join(save_path, f"truncation_comparison_{structure.value}.pdf")
        plot_all(md_filter, data_dir, save_path=save_path)
