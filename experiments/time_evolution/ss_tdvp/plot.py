"""
Script to plot the TDVP simulation results.
"""
import os
import sys

import matplotlib.pyplot as plt

import numpy.typing as npt

from pytreenet.util.plotting.plotables.standard_plottable import StandardPlottable
from pytreenet.util.plotting.plotables.multiplot import ConvergingPlottable
from pytreenet.util.experiment_util.metadata_file import MetadataFilter
from pytreenet.time_evolution.results import Results
from pytreenet.util.plotting.configuration import (DocumentStyle,
                                                   set_size,
                                                   config_matplotlib_to_latex)
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.util.plotting.line_config import LineConfig

from sim_script import (SimParams1TDVP,
                        Order)

def extract_data_and_time(res: Results,
                          params: SimParams1TDVP
                          ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Extracts the data and time from the simulation results.
    """
    errs = res.operator_result("error", realise=True)
    times = res.times()
    return times, errs

def cretae_line_config(order: Order) -> LineConfig:
    """
    Creates a line config for the given order.
    """
    if order is Order.FIRST:
        return LineConfig(label="First Order", color="blue", linestyle="-")
    else:
        return LineConfig(label="Second Order", color="orange", linestyle="--")

def load_data_err_t(md_filter: MetadataFilter,
              directory_path: str
              ) -> tuple[ConvergingPlottable, ConvergingPlottable]:
    """
    Loads the data for the err vs. t plottables for the TDVP simulation results.
    """
    out = []
    for order in Order:
        md_filter.change_criterium("order", order.value)
        params_res = md_filter.load_valid_results_and_parameters(directory_path,
                                                                 SimParams1TDVP)
        ord_std_pltbs: list[StandardPlottable] = []
        for params, res in params_res:
            pltb = StandardPlottable.from_simulation_result(res,
                                                            params,
                                                            extract_data_and_time)
            ord_std_pltbs.append(pltb)
        conv_pltb = ConvergingPlottable.from_multiple_standards(ord_std_pltbs,
                                                                conv_param="bond_dim")
        conv_pltb.line_config = cretae_line_config(order)
        out.append(conv_pltb)
    return tuple(out)

def plot_err_t(md_filter: MetadataFilter,
           directory_path: str
           ) -> None:
    """
    Plots the err vs. t for the TDVP simulation results.
    """
    doc_style = DocumentStyle.THESIS
    config_matplotlib_to_latex(doc_style)
    size = set_size(doc_style)
    conv_pltbs = load_data_err_t(md_filter, directory_path)
    
    fig, ax = plt.subplots(figsize=size)
    for conv_pltb in conv_pltbs:
        conv_pltb.plot_on_axis(ax)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Error")
    ax.set_xscale("log")
    ax.legend()

    save_path = os.path.join(directory_path, "plots")
    save_path = os.path.join(save_path, "err_vs_t.pdf")
    plt.savefig(save_path, bbox_inches="tight", format="pdf")

def load_bd_vs_rt(md_filter: MetadataFilter,
              directory_path: str
              ) -> tuple[StandardPlottable, StandardPlottable]:
    """
    Loads the data for the bond dimension vs. runtime plottables for
    the 1TDVP simulation results.
    """
    out = []
    for order in Order:
        md_filter.change_criterium("order", order.value)
        desired = ["bond_dim", "simulation_time"]
        attrs = md_filter.load_valid_attributes(directory_path,
                                                desired_keys=desired)
        line_config = cretae_line_config(order)
        std_pltb = StandardPlottable.create_empty(line_config=line_config)
        for attr in attrs:
            std_pltb.add_point(attr[desired[0]], attr[desired[1]])
        std_pltb.sort_by_x()
        out.append(std_pltb)
    return tuple(out)

def plot_bd_vs_rt(md_filter: MetadataFilter,
              directory_path: str
              ) -> None:
    """
    Plots the bond dimension vs. runtime for the TDVP simulation results.
    """
    doc_style = DocumentStyle.THESIS
    config_matplotlib_to_latex(doc_style)
    size = set_size(doc_style)
    std_pltbs = load_bd_vs_rt(md_filter, directory_path)
    
    fig, ax = plt.subplots(figsize=size)
    for std_pltb in std_pltbs:
        std_pltb.plot_on_axis(ax)
    ax.set_xlabel("Bond Dimension")
    ax.set_ylabel("Runtime (s)")
    ax.set_yscale("log")
    ax.legend()

    save_path = os.path.join(directory_path, "plots")
    save_path = os.path.join(save_path, "bd_vs_rt.pdf")
    plt.savefig(save_path, bbox_inches="tight", format="pdf")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <data_directory>")
        sys.exit(1)

    data_dir = sys.argv[1]
    
    md_filter = MetadataFilter()
    md_filter.change_criterium("structure", TTNStructure.MPS.value)
    md_filter.change_criterium("system_size", 14)
    md_filter.change_criterium("ext_magn", 0.5)
    md_filter.change_criterium("time_step_size", 0.1)
    plot_err_t(md_filter, data_dir)
    plot_bd_vs_rt(md_filter, data_dir)
