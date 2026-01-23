"""
This module contains functions to plot the results of the 2D nearest neighbour
experiments.
"""
import os
from enum import Enum
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from pytreenet.util.plotting.plotables.standard_plottable import StandardPlottable
from pytreenet.util.experiment_util.metadata_file import MetadataFilter
from pytreenet.time_evolution.results import Results
from pytreenet.util.plotting.line_config import LineConfig
from pytreenet.ttno.ttno_class import TTNOFinder
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.time_evolution.results import Results
from pytreenet.util.plotting.configuration import (DocumentStyle,
                                                   set_size,
                                                   config_matplotlib_to_latex)

from sim_script import (TwoDimParams, ModelKind)

class BondDimensionKind(Enum):
    MAX_BOND_DIMENSION = "max_bond_dim"
    AVERAGE_BOND_DIMENSION = "avg_bond_dim"

def load_bond_dimensions(dir_path: str,
                         md_filter: MetadataFilter,
                         desired_op: BondDimensionKind
                         ) -> StandardPlottable:
    """
    Loads bond dimension data from the specified directory based on the given
    metadata filter.

    Args:
        dir_path (str): The directory path where the bond dimension files are stored.
        md_filter (MetadataFilter): The metadata filter to select specific files.
        desired_op (BondDimensionKind): The kind of bond dimension to extract.

    Returns:
        StandardPlottable: A plottable object containing the bond dimension data.
    """
    results: list[Results] = md_filter.load_valid_results(dir_path)
    plottable = StandardPlottable(np.asarray([]),
                                  np.asarray([]))
    for res in results:
        bd = res.operator_result(desired_op.value,
                                 realise=True)
        sys_size = res.operator_result("sys_size",
                                       realise=True)
        plottable.add_point(float(sys_size[0]), float(bd[0]))
    return plottable

def get_line_config(finder: TTNOFinder) -> LineConfig:
    """
    Get the line configuration for the given TTNO finder.

    Args:
        finder (TTNOFinder): The TTNOFinder enum value.
    
    Returns:
        LineConfig: The line configuration for plotting.
    """
    out_kwargs = {"marker_size": 2, "linewidth": 0.5, "linestyle": ""}
    if finder == TTNOFinder.SGE:
        out_kwargs.update({"color": "blue", "marker": "o", "label": "Combined"})
    elif finder == TTNOFinder.SGE_PURE:
        out_kwargs.update({"color": "orange", "marker": "s", "label": "SGE"})
    elif finder == TTNOFinder.BIPARTITE:
        out_kwargs.update({"color": "green", "marker": "^", "label": "Bipartite"})
    else:
        raise ValueError(f"Invalid finder: {finder}")
    line_config = LineConfig(**out_kwargs)
    return line_config

def plot_bond_dimensions(dir_path: str,
                         output_path: str,
                         model_kind: ModelKind
                         ):
    """
    Plots the bond dimensions for different TTNO finders and TTN structures.

    Args:
        dir_path (str): The directory path where the bond dimension files are stored.
        output_path (str): The file path to save the generated plot.
        model_kind (ModelKind): The model kind to filter the data.
    """
    finders = [TTNOFinder.SGE,
               TTNOFinder.SGE_PURE,
               TTNOFinder.BIPARTITE]
    ttn_structures = [TTNStructure.MPS,
                      TTNStructure.BINARY,
                      TTNStructure.FTPS]
    config_matplotlib_to_latex(DocumentStyle.THESIS)
    size = set_size(DocumentStyle.THESIS,
                    subplots=(1, 3))
    size = (size[0], size[1] * 2)
    fig_avg, axes_avg = plt.subplots(1, 3, figsize=size, sharey=True)
    fig_max, axes_max = plt.subplots(1, 3, figsize=size, sharey=True)
    for finder in finders:
        for i, ttn_structure in enumerate(ttn_structures):
            md_filter = MetadataFilter()
            md_filter.change_criteria({"model": model_kind.value,
                                       "finder": finder.value,
                                       "ttn_structure": ttn_structure.value})
            config = get_line_config(finder)
            pltb_avg = load_bond_dimensions(dir_path,
                                            md_filter,
                                            BondDimensionKind.AVERAGE_BOND_DIMENSION)
            pltb_avg.line_config = config
            ax_avg = axes_avg[i]
            pltb_avg.plot_on_axis(ax_avg)
            pltb_max = load_bond_dimensions(dir_path,
                                            md_filter,
                                            BondDimensionKind.MAX_BOND_DIMENSION)
            pltb_max.line_config = config
            ax_max = axes_max[i]
            pltb_max.plot_on_axis(ax_max)
            if i == len(ttn_structures) - 1:
                ax_avg.legend()
                ax_max.legend()
            if i == 0:
                ax_avg.set_ylabel("avg bd")
                ax_max.set_ylabel("max bd")
            ax_avg.set_xlabel("$L$")
            ax_max.set_xlabel("$L$")
            ax_avg.grid(True)
            ax_max.grid(True)
    fig_avg.tight_layout()
    fig_max.tight_layout()
    avg_file_name = f"avgbd_{model_kind.value.lower()}.pdf"
    avg_file_path = os.path.join(output_path, avg_file_name)
    fig_avg.savefig(avg_file_path, format="pdf")
    max_file_name = f"maxbd_{model_kind.value.lower()}.pdf"
    max_file_path = os.path.join(output_path, max_file_name)
    fig_max.savefig(max_file_path, format="pdf")

def main():
    """
    Main function to parse arguments and plot bond dimensions.
    """
    parser = ArgumentParser()
    parser.add_argument("dir_path", type=str,
                        help="Directory path where bond dimension files are stored.")
    dir_path = vars(parser.parse_args())["dir_path"]
    output_path = os.path.join(dir_path, "plots")
    print("Output path:", output_path)
    os.makedirs(output_path, exist_ok=True)
    for model_kind in [ModelKind.ISING, ModelKind.XXZ]:
        plot_bond_dimensions(dir_path,
                             output_path,
                             model_kind)

if __name__ == "__main__":
    main()