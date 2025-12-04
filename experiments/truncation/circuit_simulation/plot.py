"""
Script to plot results from circuit simulation experiments with varying truncation levels.
"""

import os
import sys

import matplotlib.pyplot as plt

from pytreenet.ttns.ttns_ttno.application import ApplicationMethod
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.time_evolution.results import Results
from pytreenet.util.experiment_util.metadata_file import MetadataFilter
from pytreenet.util.plotting.plotables.standard_plottable import (StandardPlottable,
                                                                  combine_equivalent_standard_plottables)
from pytreenet.util.plotting.plotables.multiplot import multiple_conv_from_multiple_standard
from pytreenet.util.plotting.line_config import LineConfig, StyleMapping, StyleOption
from pytreenet.util.plotting.configuration import (set_size,
                                                   config_matplotlib_to_latex,
                                                   DocumentStyle)

from sim_script import (RES_IDS,
                        CircuitSimParams)

def _get_result(result: Results,
                result_identifier: str):
    """
    Helper function to extract a specific result from the Results object.

    Args:
        result (Results): The Results object containing simulation results.
        result_identifier (str): The identifier for the specific result to extract.
    
    Returns:
        The extracted result corresponding to the given identifier.
    """
    max_bd = result.operator_result(RES_IDS[0])
    other_res = result.operator_result(result_identifier)
    return max_bd, other_res

def load_single_results(data_path: str,
                        md_filter: MetadataFilter,
                        result_identifier: str
                        ) -> list[StandardPlottable]:
    """
    Load the individual results for single bond dimensions.

    Args:
        data_path (str): Path to the data directory.
        md_filter (MetadataFilter): Metadata filter to select the desired results.
        result_identifier (str): Identifier for the specific result to load.
    """
    single_datas = md_filter.load_valid_results_and_parameters(data_path,
                                                               parameter_class=CircuitSimParams,
                                                               allow_non_exist=True)
    single_runs: list[StandardPlottable] = []
    ign_keys = {"min_bond_dim", "max_bond_dim"}
    for params, single_data in single_datas:
        pltb = StandardPlottable.from_simulation_result(single_data,
                                                        params,
                                                        lambda res: _get_result(res, result_identifier))
        for other_pltb in single_runs:
            if pltb.assoc_equal(other_pltb, ignored_keys=ign_keys):
                other_pltb.add_point(pltb.x[0], pltb.y[0])
                break
        else:
            single_runs.append(pltb)
    for s_run in single_runs:
        s_run.sort_by_x()
        s_run.assoc_params.pop("min_bond_dim", None)
        s_run.assoc_params.pop("max_bond_dim", None)
    return single_runs

def average_seeded_result(pltbs: list[StandardPlottable]
                          ) -> list[StandardPlottable]:
    """
    Averages results over different random seeds.

    Args:
        pltbs (list[StandardPlottable]): List of StandardPlottable objects
            with different parameters and the parameter "seed".
        
    Returns:
        list[StandardPlottable]: List of StandardPlottable objects with same
            paramters except for "seed", averaged over the different seeds.
    """
    convs = multiple_conv_from_multiple_standard(pltbs,
                                                conv_param="seed",
                                                allow_x_mismatch=True)
    return [conv.average_results() for conv in convs]

def build_style_mapping() -> StyleMapping:
    """
    Builds a StyleMapping object for plotting based on simulation parameters.

    Returns:
        StyleMapping: The style mapping for plotting.
    """
    sm = StyleMapping()
    sm.add_mapping("appl_method", StyleOption.COLOR,
                   {ApplicationMethod.DENSITY_MATRIX.value: "tab:orange",
                    ApplicationMethod.HALF_DENSITY_MATRIX.value: "tab:green",
                    ApplicationMethod.SRC.value: "tab:red",
                    ApplicationMethod.ZIPUP.value: "tab:purple",
                    ApplicationMethod.DIRECT_TRUNCATE.value: "tab:blue"})
    sm.add_mapping("appl_method", StyleOption.MARKER,
                   {ApplicationMethod.DENSITY_MATRIX.value: "o",
                    ApplicationMethod.HALF_DENSITY_MATRIX.value: "s",
                    ApplicationMethod.SRC.value: "^",
                    ApplicationMethod.ZIPUP.value: "D",
                    ApplicationMethod.DIRECT_TRUNCATE.value: "v"})
    # sm.add_mapping("ttn_structure", StyleOption.LINESTYLE,
    #                  {TTNStructure.TSTAR.value: "dashdot",
    #                   TTNStructure.MPS.value: "dashed",
    #                   TTNStructure.BINARY.value: "dotted"})
    labels_method = {ApplicationMethod.DENSITY_MATRIX.value: "DM",
                     ApplicationMethod.HALF_DENSITY_MATRIX.value: "CBC",
                     ApplicationMethod.SRC.value: "SRC",
                     ApplicationMethod.ZIPUP.value: "ZipUp",
                     ApplicationMethod.DIRECT_TRUNCATE.value: "Direct"}
    # labels_structure = {TTNStructure.TSTAR.value: "MPS-LTS",
    #                     TTNStructure.MPS.value: "MPS",
    #                     TTNStructure.BINARY.value: "LOTS"}
    for key, label in labels_method.items():
        sm.set_label("appl_method", key, label)
    # for key, label in labels_structure.items():
    #     sm.set_label("ttn_structure", key, label)
    return sm

def plot_truncation_results(data_path: str,
                            md_filter: MetadataFilter,
                            save_path: str,
                            file_name: str = "circ_sim.pdf"):
    """
    Plots the truncation results from the circuit simulation experiments.

    Args:
        data_path (str): Path to the data directory.
        md_filter (MetadataFilter): Metadata filter to select the desired results.
        save_path (str): Path to save the generated plots.
    """
    os.makedirs(save_path, exist_ok=True)
    style_mapping = build_style_mapping()
    pltbs = []
    for res_id in RES_IDS[1:]:
        single_pltbs = load_single_results(data_path,
                                          md_filter,
                                          res_id)
        avg_pltbs = average_seeded_result(single_pltbs)
        for pltb in avg_pltbs:
            pltb.apply_style_mapping(style_mapping)
        pltbs.append(avg_pltbs)
    rt_vs_err: list[StandardPlottable] = []
    ms_vs_err: list[StandardPlottable] = []
    for err_pltb in pltbs[0]:
        for rt_pltb in pltbs[1]:
            if err_pltb.assoc_equal(rt_pltb):
                new = combine_equivalent_standard_plottables(rt_pltb,
                                                             err_pltb)
                rt_vs_err.append(new)
                break
        for ms_pltb in pltbs[2]:
            if err_pltb.assoc_equal(ms_pltb):
                new = combine_equivalent_standard_plottables(ms_pltb,
                                                             err_pltb)
                ms_vs_err.append(new)
                break
    config_matplotlib_to_latex(style=DocumentStyle.PRTWO_COLUMN)
    size = set_size(width=DocumentStyle.PRONE_COLUMN,
                    subplots=(1, 2))
    fig, axs = plt.subplots(1, 2, figsize=(size[0], 1.2*size[1]))
    for pltb in rt_vs_err:
        pltb.line_config.marker_size = 3
        pltb.plot_on_axis(axs[0])
    for pltb in ms_vs_err:
        pltb.line_config.marker_size = 3
        pltb.plot_on_axis(axs[1])
    axs[0].set_ylim(bottom=1e-2, top=2e0)
    axs[1].set_ylim(bottom=1e-2, top=2e0)
    axs[1].set_xlim(left=1e3,right=1e6+1e5)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Runtime (s)")
    axs[0].set_ylabel("Error")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Max Size")
    axs[1].set_ylabel("Error")
    style_mapping.apply_legend(axs[1])
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(save_path, file_name)
    fig.savefig(fig_path, format="pdf")
    plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <data_directory>")
        sys.exit(1)
    data_directory = sys.argv[1]
    save_directory = os.path.join(data_directory, "plots")
    md_filter = MetadataFilter()
    md_filter.add_to_criterium("appl_method", [ApplicationMethod.DENSITY_MATRIX.value,
                                                ApplicationMethod.SRC.value,
                                                ApplicationMethod.ZIPUP.value,
                                                ApplicationMethod.DIRECT_TRUNCATE.value,
                                                ApplicationMethod.HALF_DENSITY_MATRIX.value])
    md_filter.add_to_criterium("num_circuit_repeats", [3])
    md_filter.add_to_criterium("seed", [1234, 4321])
    md_filter.add_to_criterium("min_bond_dim", list(range(5, 201, 10)))
    ttns_structures = [TTNStructure.TSTAR,
                       TTNStructure.MPS,
                       TTNStructure.BINARY]
    for structure in ttns_structures:
        md_filter.change_criterium("ttn_structure", [structure.value])
        plot_truncation_results(data_directory,
                                md_filter,
                                save_directory,
                                file_name=f"circ_sim_{structure.value}.pdf")
