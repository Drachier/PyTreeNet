"""
Plots the results from the addition comparison experiments.
"""
import sys
import os
from itertools import product

from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt

from pytreenet.util.plotting.configuration import (DocumentStyle,
                                                   set_size,
                                                   config_matplotlib_to_latex)
from pytreenet.util.experiment_util.metadata_file import MetadataFilter
from pytreenet.core.addition.addition import AdditionMethod
from pytreenet.time_evolution.results import Results
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.util.plotting.plotables.plottable_2d import (Plottable2D,
                                                            average,
                                                            NormalisationMethod)

from sim_script import (AdditionComparisonParams,
                        RES_IDS)

def extract_bd_vs_err(result: Results
                      ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Extracts the bond dimension vs. error data from the simulation results.

    Args:
        result (Results): The simulation results containing error data.

    Returns:
        tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]: Three arrays containing
        cache bond dimensions, addition bond dimensions, and corresponding errors.
    """
    cache_bds = result.operator_result(RES_IDS[0])
    add_bds = result.operator_result(RES_IDS[1])
    errors = result.operator_result(RES_IDS[2])
    return cache_bds, add_bds, errors


def extract_bd_vs_runtime(result: Results
                          ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Extracts the bond dimension vs. runtime data from the simulation results.

    Args:
        result (Results): The simulation results containing runtime data.

    Returns:
        tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]: Three arrays containing
        cache bond dimensions, addition bond dimensions, and corresponding runtimes.
    """
    cache_bds = result.operator_result(RES_IDS[0])
    add_bds = result.operator_result(RES_IDS[1])
    runtimes = result.operator_result(RES_IDS[3])
    return cache_bds, add_bds, runtimes


def load_data(md_filter: MetadataFilter,
              directory_path: str
              ) -> tuple[Plottable2D, Plottable2D]:
    """
    Load data from the specified directory and filter it based on the provided
    metadata filter.

    Args:
        md_filter (MetadataFilter): The metadata filter to apply.
        directory_path (str): The path to the directory containing the data.

    Returns:
        tuple[Plottable2D, Plottable2D]: A tuple of two Plottable2D objects representing
            bond dimension vs. error and bond dimension vs. runtime data.
    """
    params_results = md_filter.load_valid_results_and_parameters(directory_path,
                                                                 parameter_class=AdditionComparisonParams,
                                                                 allow_non_exist=True)
    pltbs: tuple[list[Plottable2D],
                    list[Plottable2D]] = ([],[])

    for params, results in params_results:
        bd_vs_err = Plottable2D.from_simulation_result(results,
                                                       params,
                                                       lambda r, p: extract_bd_vs_err(r))
        bd_vs_runtime = Plottable2D.from_simulation_result(results,
                                                           params,
                                                           lambda r, p: extract_bd_vs_runtime(r))
        pltbs[0].append(bd_vs_err)
        pltbs[1].append(bd_vs_runtime)
    out = (average(pltbs[0]), average(pltbs[1]))
    return out

def plot(md_filter: MetadataFilter,
             directory_path: str,
             save_path: str | None = None
             ) -> None:
    """
    Plots the error and runtime for a given metadata filter.

    Args:
        md_filter (MetadataFilter): The metadata filter to apply.
        directory_path (str): The path to the directory containing the data.
        save_path (str | None, optional): The path to save the plots. If None,
            the plots will be displayed instead of being saved. Defaults to None.
    """
    data = load_data(md_filter, directory_path)

    config_matplotlib_to_latex(style=DocumentStyle.THESIS)
    size = set_size(width=DocumentStyle.THESIS, subplots=(1, 2))
    fig, axes = plt.subplots(1, 2, figsize=size)
    for ind, axdat in enumerate(zip(axes, data)):
        ax, dat = axdat
        print(save_path, dat.vals)
        dat.plot_on_axis(ax, norm_method=NormalisationMethod.LOG)
        ax.set_xlabel(r"$\chi_{\mathcal{S}}$")
        ax.set_ylabel(r"$\chi_{\text{add}}$")
        if ind == 0:
            fig.colorbar(ax.images[-1], ax=ax, label="Error")
        else:
            fig.colorbar(ax.images[-1], ax=ax, label="Runtime (s)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <data_directory>")
        sys.exit(1)

    data_dir = sys.argv[1]

    # Define the structures and corresponding system sizes
    structures = [TTNStructure.MPS,
                  TTNStructure.FTPS,
                  TTNStructure.BINARY,
                  TTNStructure.TSTAR
                  ]
    sys_sizes = {"mps": 100, "ftps": 10, "binary": 6, "tstar": 33}
    num_ads = [2]
    methods = [AdditionMethod.HALF_DENSITY_MATRIX, AdditionMethod.SRC]

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Create one plot per structure showing all methods and N values
    for method, nads, structure in product(methods, num_ads, structures):
        md_filter = MetadataFilter()
        md_filter.add_to_criterium("addition_method", [method.value])
        md_filter.add_to_criterium("num_additions", [nads])
        md_filter.add_to_criterium("structure", structure.value)
        md_filter.add_to_criterium("sys_size", sys_sizes[structure.value])
        md_filter.add_to_criterium("phys_dim", 2)  # Assuming phys_dim is fixed at 2 for all experiments
        save_path = os.path.join(plots_dir,
                                 f"addition_comparison_{method.value}_{structure.value}_{nads}.pdf")
        plot(md_filter, data_dir, save_path=save_path)
        print(f"Created plot: {save_path}")
