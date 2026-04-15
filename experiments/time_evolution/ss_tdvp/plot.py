"""
Script to plot the TDVP simulation results.
"""
import os
import sys

import matplotlib.pyplot as plt
from h5py import File
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
    errs = res.operator_result("error", realise=True)[1:]
    times = res.times()[1:]
    return times, errs

def create_line_config(order: Order) -> LineConfig:
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
                                                                 SimParams1TDVP,
                                                                 allow_non_exist=True)
        ord_std_pltbs: list[StandardPlottable] = []
        for params, res in params_res:
            pltb = StandardPlottable.from_simulation_result(res,
                                                            params,
                                                            extract_data_and_time)
            ord_std_pltbs.append(pltb)
        conv_pltb = ConvergingPlottable.from_multiple_standards(ord_std_pltbs,
                                                                conv_param="bond_dim")
        conv_pltb.line_config = create_line_config(order)
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
    ax.set_yscale("log")
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
                                                desired_keys=desired,
                                                allow_non_exist=True)
        line_config = create_line_config(order)
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

def _custom_loader_dt_finalerror(file: File) -> tuple[float, float]:
    """
    Custom loader function to extract the time step size and final error from
    the HDF5 file.
    """
    dt = file.attrs["time_step_size"]
    error = file["error"][-1]
    return dt, error

def load_dt_vs_final_error(md_filter: MetadataFilter,
                           directory_path: str
                           ) -> tuple[StandardPlottable, StandardPlottable]:
    """
    Loads the data for the time step size vs. final error plottables for
    the 1TDVP simulation results.
    """
    out = []
    for order in Order:
        md_filter.change_criterium("order", order.value)
        results = md_filter.load_custom(directory_path,
                                        _custom_loader_dt_finalerror,
                                        allow_non_exist=True)
        print(results)
        line_config = create_line_config(order)
        std_pltb = StandardPlottable.create_empty(line_config=line_config)
        for dt, error in results:
            std_pltb.add_point(dt, error)
        std_pltb.sort_by_x()
        out.append(std_pltb)
    return tuple(out)

def plot_dt_vs_final_error(md_filter: MetadataFilter,
                           directory_path: str
                           ) -> None:
    """
    Plots the time step size vs. final error for the TDVP simulation results.
    """
    doc_style = DocumentStyle.THESIS
    config_matplotlib_to_latex(doc_style)
    size = set_size(doc_style)
    std_pltbs = load_dt_vs_final_error(md_filter, directory_path)

    fig, ax = plt.subplots(figsize=size)
    for std_pltb in std_pltbs:
        std_pltb.plot_on_axis(ax)
    ax.set_xlabel("Time Step Size")
    ax.set_ylabel("Final Error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    save_path = os.path.join(directory_path, "plots")
    save_path = os.path.join(save_path, "dt_vs_final_error.pdf")
    plt.savefig(save_path, bbox_inches="tight", format="pdf")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <data_directory>")
        sys.exit(1)

    data_dir = sys.argv[1]
    
    md_filter = MetadataFilter()
    md_filter.change_criterium("structure", TTNStructure.TSTAR.value)
    md_filter.change_criterium("system_size", 3)
    md_filter.change_criterium("ext_magn", 0.5)
    md_filter.change_criterium("time_step_size", 0.01)
    plot_err_t(md_filter, data_dir)
    plot_bd_vs_rt(md_filter, data_dir)
    md_filter.remove_criterium("time_step_size")
    md_filter.change_criterium("bond_dim", 64)
    plot_dt_vs_final_error(md_filter, data_dir)
