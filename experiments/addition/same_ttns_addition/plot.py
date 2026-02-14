"""
Plots the results from the addition comparison experiments.
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
from pytreenet.core.addition.addition import AdditionMethod
from pytreenet.time_evolution.results import Results
from pytreenet.util.plotting.line_config import LineConfig, StyleMapping, StyleOption
from pytreenet.special_ttn.special_states import TTNStructure

from sim_script import (AdditionComparisonParams,
                        RES_IDS)


def method_name(params: AdditionComparisonParams | AdditionMethod) -> str:
    """
    Generates a descriptive name for the addition method used in the simulation.

    Args:
        params (AdditionComparisonParams): The parameters of the simulation.

    Returns:
        str: A descriptive name for the addition method.
    """
    method_names = {
        AdditionMethod.DIRECT_TRUNCATE: "Direct+Truncate",
        AdditionMethod.DENSITY_MATRIX: "Density Matrix",
        AdditionMethod.HALF_DENSITY_MATRIX: "Half Density Matrix",
        AdditionMethod.SRC: "SRC"
    }
    if isinstance(params, AdditionComparisonParams):
        method = params.addition_method
    else:
        method = params
    return method_names.get(method, "Unknown Method")

def method_colour(params: AdditionComparisonParams | AdditionMethod) -> str:
    """
    Assigns a color to the addition method for plotting purposes.

    Args:
        params (AdditionComparisonParams): The parameters of the simulation.

    Returns:
        str: A color code for the addition method.
    """
    if isinstance(params, AdditionComparisonParams):
        method = params.addition_method
    else:
        method = params
    if method is AdditionMethod.DIRECT_TRUNCATE:
        return "tab:orange"
    if method is AdditionMethod.DENSITY_MATRIX:
        return "tab:green"
    if method is AdditionMethod.HALF_DENSITY_MATRIX:
        return "tab:red"
    if method is AdditionMethod.SRC:
        return "tab:purple"
    return "tab:gray"


def method_marker(params: AdditionComparisonParams | AdditionMethod) -> str:
    """
    Assigns a marker to the addition method for plotting purposes.

    Args:
        params (AdditionComparisonParams): The parameters of the simulation.

    Returns:
        str: A marker style for the addition method.
    """
    markers = {
        AdditionMethod.DIRECT_TRUNCATE: "s",
        AdditionMethod.DENSITY_MATRIX: "^",
        AdditionMethod.HALF_DENSITY_MATRIX: "v",
        AdditionMethod.SRC: "D"
    }
    if isinstance(params, AdditionComparisonParams):
        method = params.addition_method
    else:
        method = params
    return markers.get(method, "o")

def num_additions_linestyle(num_additions: int) -> str:
    """
    Assigns a linestyle based on the number of additions.

    Args:
        num_additions (int): The number of additions (N).

    Returns:
        str: A linestyle for plotting.
    """
    linestyles = {
        2: "-",
        5: "--",
        10: ":"
    }
    return linestyles.get(num_additions, "-")


def build_lineconfig(params: AdditionComparisonParams) -> LineConfig:
    """
    Builds a LineConfig object for plotting based on the simulation parameters.

    Args:
        params (AdditionComparisonParams): The parameters of the simulation.

    Returns:
        LineConfig: The configuration for plotting lines.
    """
    lc = LineConfig()
    lc.color = method_colour(params)
    lc.label = f"{method_name(params)} (N={params.num_additions})"
    lc.marker = method_marker(params)
    lc.linestyle = num_additions_linestyle(params.num_additions)
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
    Extracts the runtime vs. error data from the simulation results.

    Args:
        result (Results): The simulation results containing error and runtime data.

    Returns:
        tuple[NDArray[np.floating], NDArray[np.floating]]: Two arrays containing
        runtimes and corresponding errors.
    """
    runtimes = np.cumsum(result.operator_result(RES_IDS[2]))
    errors = result.operator_result(RES_IDS[1])
    return runtimes, errors


def load_data(md_filter: MetadataFilter,
              directory_path: str
              ) -> dict[tuple[str, int], tuple[StandardPlottable, StandardPlottable, StandardPlottable]]:
    """
    Load data from the specified directory and filter it based on the provided
    metadata filter.

    Args:
        md_filter (MetadataFilter): The metadata filter to apply.
        directory_path (str): The path to the directory containing the data.

    Returns:
        dict[tuple[str, int], tuple[StandardPlottable, StandardPlottable, StandardPlottable]]:
            A dictionary mapping (method_name, num_additions) tuples to tuples of plottables
            for bond dimension vs. error, bond dimension vs. runtime, and runtime vs. error.
    """
    params_results = md_filter.load_valid_results_and_parameters(directory_path,
                                                                 parameter_class=AdditionComparisonParams)
    out: dict[tuple[str, int],
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
        key = (m_name, params.num_additions)
        if key not in out:
            out[key] = ([], [], [])
        out[key][0].append(bd_vs_err)
        out[key][1].append(bd_vs_runtime)

    # Create converging plottables and average them
    actual_out: dict[tuple[str, int], tuple[StandardPlottable,
                                            StandardPlottable, StandardPlottable]] = {}
    for key, (bd_vs_err_list, bd_vs_runtime_list, _) in out.items():
        conv_plots = (ConvergingPlottable.from_multiple_standards(bd_vs_err_list,
                                                                  ignored_keys={'seed'}),
                      ConvergingPlottable.from_multiple_standards(bd_vs_runtime_list,
                                                                  ignored_keys={'seed'}))
        stand_plts = (conv_plots[0].average_results(),
                      conv_plots[1].average_results())
        comb_plot = combine_equivalent_standard_plottables(
            stand_plts[1], stand_plts[0])
        actual_out[key] = (stand_plts[0], stand_plts[1], comb_plot)

    return actual_out


def build_style_mapping(md_filter: MetadataFilter) -> StyleMapping:
    """
    Build a StyleMapping for the legend based on the metadata filter.

    Args:
        md_filter (MetadataFilter): The metadata filter containing the parameter values.

    Returns:
        StyleMapping: The constructed StyleMapping for legend rendering.
    """
    style_mapping = StyleMapping()
    add_methods = md_filter.get_criterium("addition_method")
    add_methods = [AdditionMethod(method) for method in add_methods]
    style_mapping.add_mapping("addition_method",
                              StyleOption.COLOR,
                              {method.value: method_colour(method) for method in add_methods})
    style_mapping.add_mapping("addition_method",
                              StyleOption.MARKER,
                              {method.value: method_marker(method) for method in add_methods})
    for method in add_methods:
        style_mapping.set_label("addition_method",
                                method.value,
                                method_name(method))
    style_mapping.add_mapping("num_additions",
                              StyleOption.LINESTYLE,
                                {num: num_additions_linestyle(num) for num in md_filter.get_criterium("num_additions")})
    for num in md_filter.get_criterium("num_additions"):
        style_mapping.set_label("num_additions",
                                num,
                                f"N={num}")
    return style_mapping


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

    # Plot all data
    for (method, n_add), (bd_vs_err, bd_vs_runtime, runtime_vs_err) in data.items():
        bd_vs_err.plot_on_axis(axs[0],set_label=False)
        bd_vs_runtime.plot_on_axis(axs[1],set_label=False)
        runtime_vs_err.plot_on_axis(axs[2],set_label=False)

    for ax, x_label, y_label, x_log, y_log in zip(axs, x_labels, y_labels, xlog, ylog):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")

    # Build and apply custom legend using StyleMapping from MetadataFilter
    style_mapping = build_style_mapping(md_filter)
    style_mapping.apply_legend(axs[0])
    axs[0].legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <data_directory>")
        sys.exit(1)

    data_dir = sys.argv[1]

    # Define the structures and corresponding system sizes
    structures = [TTNStructure.MPS, TTNStructure.FTPS,
                  TTNStructure.BINARY, TTNStructure.TSTAR]
    sys_sizes = {"mps": 10, "ftps": 3, "binary": 2, "tstar": 5}

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Create one plot per structure showing all methods and N values
    for structure in structures:
        md_filter = MetadataFilter()
        md_filter.add_to_criterium("addition_method", [AdditionMethod.DIRECT_TRUNCATE.value,
                                                       AdditionMethod.DENSITY_MATRIX.value,
                                                       AdditionMethod.HALF_DENSITY_MATRIX.value,
                                                       AdditionMethod.SRC.value])
        md_filter.add_to_criterium("num_additions", [2, 5, 10])
        md_filter.add_to_criterium("structure", structure.value)
        md_filter.add_to_criterium("sys_size", sys_sizes[structure.value])

        save_path = os.path.join(plots_dir,
                                 f"addition_comparison_{structure.value}.pdf")
        plot_all(md_filter, data_dir, save_path=save_path)
        print(f"Created plot: {save_path}")
