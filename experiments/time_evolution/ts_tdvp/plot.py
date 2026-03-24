"""
Script to plot the integrator comparison simulation results.
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
from pytreenet.util.plotting.line_config import (LineConfig,
                                                 StyleMapping,
                                                 StyleOption)

from sim_script import SimParams2TDVP, Integrator

def extract_data_and_time(res: Results,
                          params: SimParams2TDVP
                          ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Extracts the data and time from the simulation results.
    """
    errs = res.operator_result("error", realise=True)
    times = res.times()
    return times, errs

def _integrator_label(integrator: Integrator) -> str:
    if integrator is Integrator.TWO_SITE_TDVP:
        return "2TDVP"
    return "BUG"

def _integrator_linestyle(integrator: Integrator) -> str:
    if integrator is Integrator.TWO_SITE_TDVP:
        return "-"
    return "--"

def _integrator_color(integrator: Integrator) -> str:
    if integrator is Integrator.TWO_SITE_TDVP:
        return "blue"
    return "orange"

def gen_style_mapping() -> StyleMapping:
    sm = StyleMapping()
    sm.add_mapping("integrator", StyleOption.COLOR,
                   {integ: _integrator_color(integ) for integ in Integrator})
    sm.add_mapping("integrator", StyleOption.LINESTYLE,
                   {integ: _integrator_linestyle(integ) for integ in Integrator})
    sm.set_label("integrator", Integrator.TWO_SITE_TDVP, "2TDVP")
    sm.set_label("integrator", Integrator.BUG, "BUG")
    return sm

def load_data_err_t_bond_dim(md_filter: MetadataFilter,
                             directory_path: str
                             ) -> list[ConvergingPlottable]:
    """
    Loads the data for the err vs. t plottables varying bond dimensions.
    """
    md_filter = md_filter.copy()  # Avoid modifying the original filter
    md_filter.change_criterium("rel_tol", 1e-10)
    md_filter.change_criterium("total_tol", 1e-10)
    out = []
    for integrator in Integrator:
        md_filter.change_criterium("integrator", integrator.value)
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
                                                                    conv_param="max_bond_dim")
            line_cfg = LineConfig(label=_integrator_label(integrator),
                                  linestyle=_integrator_linestyle(integrator),
                                  color=_integrator_color(integrator))
            conv_pltb.line_config = line_cfg
            out.append(conv_pltb)
    return out

def load_data_err_t_rtol(md_filter: MetadataFilter,
                         directory_path: str
                         ) -> list[ConvergingPlottable]:
    """
    Loads the data for the err vs. t plottables varying rtol values.
    """
    md_filter = md_filter.copy()  # Avoid modifying the original filter
    md_filter.change_criterium("max_bond_dim", 100)
    md_filter.change_criterium("total_tol", 1e-10)
    out = []
    for integrator in Integrator:
        md_filter.change_criterium("integrator", integrator.value)
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
                                                                    conv_param="rel_tol")
            line_cfg = LineConfig(label=_integrator_label(integrator),
                                  linestyle=_integrator_linestyle(integrator),
                                  color=_integrator_color(integrator))
            conv_pltb.line_config = line_cfg
            out.append(conv_pltb)
    return out

def load_data_err_t_atol(md_filter: MetadataFilter,
                         directory_path: str
                         ) -> list[ConvergingPlottable]:
    """
    Loads the data for the err vs. t plottables varying atol values.
    """
    md_filter = md_filter.copy()  # Avoid modifying the original filter
    md_filter.change_criterium("max_bond_dim", 100)
    md_filter.change_criterium("rel_tol", 1e-10)
    out = []
    for integrator in Integrator:
        md_filter.change_criterium("integrator", integrator.value)
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
                                                                    conv_param="total_tol")
            line_cfg = LineConfig(label=_integrator_label(integrator),
                                  linestyle=_integrator_linestyle(integrator),
                                  color=_integrator_color(integrator))
            conv_pltb.line_config = line_cfg
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
    sm = gen_style_mapping()
    sm.apply_legend(ax)
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
    sm = gen_style_mapping()
    sm.apply_legend(ax)
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
    sm = gen_style_mapping()
    sm.apply_legend(ax)
    ax.legend()

    save_path = os.path.join(directory_path, "plots")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, "err_vs_t_atol.pdf")
    plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close()

def load_bd_vs_rt(md_filter: MetadataFilter,
                  directory_path: str
                  ) -> list[StandardPlottable]:
    """
    Loads the data for the bond dimension vs. runtime plottable.
    """
    out = []
    desired = ["max_bond_dim", "simulation_time"]
    for integrator in Integrator:
        md_filter.change_criterium("integrator", integrator.value)
        attrs = md_filter.load_valid_attributes(directory_path,
                                                desired_keys=desired)
        line_cfg = LineConfig(label=_integrator_label(integrator),
                              linestyle=_integrator_linestyle(integrator))
        std_pltb = StandardPlottable.create_empty(line_config=line_cfg)
        for attr in attrs:
            std_pltb.add_point(attr[desired[0]], attr[desired[1]])
        std_pltb.sort_by_x()
        out.append(std_pltb)
    return out

def plot_bd_vs_rt(md_filter: MetadataFilter,
                  directory_path: str
                  ) -> None:
    """
    Plots the bond dimension vs. runtime for the 2TDVP simulation results.
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
    md_filter.change_criterium("system_size", 5)
    md_filter.change_criterium("ext_magn", 0)
    md_filter.change_criterium("time_step_size", 0.01)
    
    print("Plotting bond dimension dependence...")
    plot_err_t_bond_dim(md_filter, data_dir)
    print("Plotting rtol dependence...")
    plot_err_t_rtol(md_filter, data_dir)
    print("Plotting atol dependence...")
    plot_err_t_atol(md_filter, data_dir)
    print("Plotting bond dimension vs runtime...")
    plot_bd_vs_rt(md_filter, data_dir)
    print("All plots created successfully!")
