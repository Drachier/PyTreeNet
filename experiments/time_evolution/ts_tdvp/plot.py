"""
Script to plot the 2TDVP simulation results.
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

from sim_script import SimParams2TDVP

def extract_data_and_time(res: Results,
                          params: SimParams2TDVP
                          ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Extracts the data and time from the simulation results.
    """
    errs = res.operator_result("error", realise=True)
    times = res.times()
    return times, errs

def create_line_config_bond_dim(bond_dim: int) -> LineConfig:
    """
    Creates a line config for the given bond dimension.
    """
    colors = {5: "blue", 10: "green", 15: "orange", 20: "red", 
              25: "purple", 30: "brown", 50: "pink", 100: "gray"}
    color = colors.get(bond_dim, "blue")
    return LineConfig(label=f"bd={bond_dim}", color=color, linestyle="-")

def create_line_config_rtol(rtol: float) -> LineConfig:
    """
    Creates a line config for the given rtol value.
    """
    colors = {1e-6: "blue", 1e-8: "orange", 1e-10: "red"}
    linestyles = {1e-6: "-", 1e-8: "--", 1e-10: "-."}
    color = colors.get(rtol, "blue")
    linestyle = linestyles.get(rtol, "-")
    return LineConfig(label=f"rtol={rtol:.0e}", color=color, linestyle=linestyle)

def create_line_config_atol(atol: float) -> LineConfig:
    """
    Creates a line config for the given atol value.
    """
    colors = {1e-6: "blue", 1e-8: "orange", 1e-10: "red"}
    linestyles = {1e-6: "-", 1e-8: "--", 1e-10: "-."}
    color = colors.get(atol, "blue")
    linestyle = linestyles.get(atol, "-")
    return LineConfig(label=f"atol={atol:.0e}", color=color, linestyle=linestyle)

def load_data_err_t_bond_dim(md_filter: MetadataFilter,
                             directory_path: str
                             ) -> list[ConvergingPlottable]:
    """
    Loads the data for the err vs. t plottables varying bond dimensions.
    """
    out = []
    bond_dims = [5, 10, 15, 20, 25, 30, 50, 100]
    for bd in bond_dims:
        md_filter.change_criterium("bond_dim", bd)
        params_res = md_filter.load_valid_results_and_parameters(directory_path,
                                                                 SimParams2TDVP)
        bd_std_pltbs: list[StandardPlottable] = []
        for params, res in params_res:
            pltb = StandardPlottable.from_simulation_result(res,
                                                            params,
                                                            extract_data_and_time)
            bd_std_pltbs.append(pltb)
        if bd_std_pltbs:  # Only if there are results
            conv_pltb = ConvergingPlottable.from_multiple_standards(bd_std_pltbs,
                                                                    conv_param=None)
            conv_pltb.line_config = create_line_config_bond_dim(bd)
            out.append(conv_pltb)
    return out

def load_data_err_t_rtol(md_filter: MetadataFilter,
                         directory_path: str
                         ) -> list[ConvergingPlottable]:
    """
    Loads the data for the err vs. t plottables varying rtol values.
    """
    out = []
    rtol_values = [1e-6, 1e-8, 1e-10]
    for rtol in rtol_values:
        md_filter.change_criterium("rtol", rtol)
        params_res = md_filter.load_valid_results_and_parameters(directory_path,
                                                                 SimParams2TDVP)
        rtol_std_pltbs: list[StandardPlottable] = []
        for params, res in params_res:
            pltb = StandardPlottable.from_simulation_result(res,
                                                            params,
                                                            extract_data_and_time)
            rtol_std_pltbs.append(pltb)
        if rtol_std_pltbs:  # Only if there are results
            conv_pltb = ConvergingPlottable.from_multiple_standards(rtol_std_pltbs,
                                                                    conv_param=None)
            conv_pltb.line_config = create_line_config_rtol(rtol)
            out.append(conv_pltb)
    return out

def load_data_err_t_atol(md_filter: MetadataFilter,
                         directory_path: str
                         ) -> list[ConvergingPlottable]:
    """
    Loads the data for the err vs. t plottables varying atol values.
    """
    out = []
    atol_values = [1e-6, 1e-8, 1e-10]
    for atol in atol_values:
        md_filter.change_criterium("atol", atol)
        params_res = md_filter.load_valid_results_and_parameters(directory_path,
                                                                 SimParams2TDVP)
        atol_std_pltbs: list[StandardPlottable] = []
        for params, res in params_res:
            pltb = StandardPlottable.from_simulation_result(res,
                                                            params,
                                                            extract_data_and_time)
            atol_std_pltbs.append(pltb)
        if atol_std_pltbs:  # Only if there are results
            conv_pltb = ConvergingPlottable.from_multiple_standards(atol_std_pltbs,
                                                                    conv_param=None)
            conv_pltb.line_config = create_line_config_atol(atol)
            out.append(conv_pltb)
    return out

def plot_err_t_bond_dim(md_filter: MetadataFilter,
                        directory_path: str
                        ) -> None:
    """
    Plots the err vs. t for varying bond dimensions.
    """
    doc_style = DocumentStyle.THESIS
    config_matplotlib_to_latex(doc_style)
    size = set_size(doc_style)
    conv_pltbs = load_data_err_t_bond_dim(md_filter, directory_path)
    
    fig, ax = plt.subplots(figsize=size)
    for conv_pltb in conv_pltbs:
        conv_pltb.plot_on_axis(ax)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Error")
    ax.set_xscale("log")
    ax.legend()

    save_path = os.path.join(directory_path, "plots")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, "err_vs_t_bond_dim.pdf")
    plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close()

def plot_err_t_rtol(md_filter: MetadataFilter,
                    directory_path: str
                    ) -> None:
    """
    Plots the err vs. t for varying rtol values.
    """
    doc_style = DocumentStyle.THESIS
    config_matplotlib_to_latex(doc_style)
    size = set_size(doc_style)
    conv_pltbs = load_data_err_t_rtol(md_filter, directory_path)
    
    fig, ax = plt.subplots(figsize=size)
    for conv_pltb in conv_pltbs:
        conv_pltb.plot_on_axis(ax)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Error")
    ax.set_xscale("log")
    ax.legend()

    save_path = os.path.join(directory_path, "plots")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, "err_vs_t_rtol.pdf")
    plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close()

def plot_err_t_atol(md_filter: MetadataFilter,
                    directory_path: str
                    ) -> None:
    """
    Plots the err vs. t for varying atol values.
    """
    doc_style = DocumentStyle.THESIS
    config_matplotlib_to_latex(doc_style)
    size = set_size(doc_style)
    conv_pltbs = load_data_err_t_atol(md_filter, directory_path)
    
    fig, ax = plt.subplots(figsize=size)
    for conv_pltb in conv_pltbs:
        conv_pltb.plot_on_axis(ax)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Error")
    ax.set_xscale("log")
    ax.legend()

    save_path = os.path.join(directory_path, "plots")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, "err_vs_t_atol.pdf")
    plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close()

def load_bd_vs_rt(md_filter: MetadataFilter,
                  directory_path: str
                  ) -> StandardPlottable:
    """
    Loads the data for the bond dimension vs. runtime plottable.
    """
    desired = ["bond_dim", "simulation_time"]
    attrs = md_filter.load_valid_attributes(directory_path,
                                            desired_keys=desired)
    std_pltb = StandardPlottable.create_empty(line_config=LineConfig(label="2TDVP"))
    for attr in attrs:
        std_pltb.add_point(attr[desired[0]], attr[desired[1]])
    std_pltb.sort_by_x()
    return std_pltb

def plot_bd_vs_rt(md_filter: MetadataFilter,
                  directory_path: str
                  ) -> None:
    """
    Plots the bond dimension vs. runtime for the 2TDVP simulation results.
    """
    doc_style = DocumentStyle.THESIS
    config_matplotlib_to_latex(doc_style)
    size = set_size(doc_style)
    std_pltb = load_bd_vs_rt(md_filter, directory_path)
    
    fig, ax = plt.subplots(figsize=size)
    std_pltb.plot_on_axis(ax)
    ax.set_xlabel("Bond Dimension")
    ax.set_ylabel("Runtime (s)")
    ax.set_yscale("log")
    ax.legend()

    save_path = os.path.join(directory_path, "plots")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, "bd_vs_rt.pdf")
    plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close()

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
    
    print("Plotting bond dimension dependence...")
    plot_err_t_bond_dim(md_filter, data_dir)
    print("Plotting rtol dependence...")
    plot_err_t_rtol(md_filter, data_dir)
    print("Plotting atol dependence...")
    plot_err_t_atol(md_filter, data_dir)
    print("Plotting bond dimension vs runtime...")
    plot_bd_vs_rt(md_filter, data_dir)
    print("All plots created successfully!")
