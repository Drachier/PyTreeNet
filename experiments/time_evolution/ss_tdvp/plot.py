"""
Script to plot the TDVP simulation results.
"""
import os
import sys

import matplotlib.pyplot as plt
from h5py import File
import numpy.typing as npt
import numpy as np

from pytreenet.util.plotting.plotables.standard_plottable import StandardPlottable
from pytreenet.util.plotting.plotables.multiplot import ConvergingPlottable
from pytreenet.util.experiment_util.metadata_file import (MetadataFilter,
                                                          METADATAFILE_STANDARD_NAME)
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
    size = set_size(doc_style, subplots=(1,3))

    pltbs: dict[TTNStructure, tuple[ConvergingPlottable, ConvergingPlottable]] = {}
    for struct in [TTNStructure.MPS, TTNStructure.BINARY, TTNStructure.TSTAR]:
        md_filter.change_criterium("structure", struct.value)
        conv_pltbs = load_data_err_t(md_filter, directory_path)
        pltbs[struct] = conv_pltbs
    
    fig, ax = plt.subplots(1, 3, figsize=(size[0], 1.25*size[1]))
    for i, (struct, conv_pltbs) in enumerate(pltbs.items()):
        for conv_pltb in conv_pltbs:
            conv_pltb.plot_on_axis(ax[i])
    for a in ax:
        a.set_xlabel(r"$t$")
        a.set_yscale("log")
    ax[0].set_ylabel("Error")
    fig.tight_layout()

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
    size = set_size(doc_style, subplots=(1,3))

    pltbs: dict[TTNStructure, tuple[StandardPlottable, StandardPlottable]] = {}
    for struct in [TTNStructure.MPS, TTNStructure.BINARY, TTNStructure.TSTAR]:
        md_filter.change_criterium("structure", struct.value)
        pltbs[struct] = load_bd_vs_rt(md_filter, directory_path)

    fig, ax = plt.subplots(1, 3, figsize=size)
    for i, (struct, (std_pltb1, std_pltb2)) in enumerate(pltbs.items()):
        std_pltb1.plot_on_axis(ax[i])
        std_pltb2.plot_on_axis(ax[i])
        ax[i].set_xlabel("Bond Dimension")
        ax[i].set_yscale("log")
    ax[0].set_ylabel("Runtime (s)")
    ax[2].legend()

    save_path = os.path.join(directory_path, "plots")
    save_path = os.path.join(save_path, "bd_vs_rt.pdf")
    plt.savefig(save_path, bbox_inches="tight", format="pdf")

def _custom_loader_dt_finalerror(file: File,
                                 mdfilter: MetadataFilter) -> tuple[float, float]:
    """
    Custom loader function to extract the time step size and final error from
    the HDF5 file.
    """
    time_steps_key = "time_step_size"
    if time_steps_key in file.attrs:
        dt = file.attrs[time_steps_key]
    else:
        # In this case we load the time step size from the metadata
        filename = os.path.basename(file.filename)  # Get the filename from the file path
        hash_id = filename.split(".")[0]  # Assuming the filename is in the format "hash_id.h5"
        run_metadata = mdfilter.md_dict[hash_id]
        dt = run_metadata[time_steps_key]
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
                                        lambda file: _custom_loader_dt_finalerror(file, md_filter),
                                        allow_non_exist=True)
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
    size = set_size(doc_style, subplots=(1,3))

    pltbs: dict[TTNStructure, tuple[StandardPlottable, StandardPlottable]] = {}
    for struct in [TTNStructure.MPS, TTNStructure.BINARY, TTNStructure.TSTAR]:
        md_filter.change_criterium("structure", struct.value)
        pltbs[struct] = load_dt_vs_final_error(md_filter, directory_path)

    fig, ax = plt.subplots(1, 3, figsize=(size[0], 1.25*size[1]))
    for i, (struct, (std_pltb1, std_pltb2)) in enumerate(pltbs.items()):
        std_pltb1.plot_on_axis(ax[i])
        std_pltb2.plot_on_axis(ax[i])
        ax[i].set_xlabel(r"Time Step Size $\Delta t$")
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
    ax[0].set_ylabel("Final Error")
    ax[2].legend()
    plt.tight_layout()

    save_path = os.path.join(directory_path, "plots")
    save_path = os.path.join(save_path, "dt_vs_final_error.pdf")
    plt.savefig(save_path, bbox_inches="tight", format="pdf")

def load_accerr_vs_bd(md_filter: MetadataFilter,
                           directory_path: str
                           ) -> tuple[StandardPlottable, StandardPlottable]:
    """
    Loads the data for the bond dimension vs. accumulated error plottables for
    the 1TDVP simulation results.
    """
    out = []
    for order in Order:
        md_filter.change_criterium("order", order.value)
        params_res = md_filter.load_valid_results_and_parameters(directory_path,
                                                                 SimParams1TDVP,
                                                                 allow_non_exist=True)
        lc = create_line_config(order)
        avg_pltb = StandardPlottable.create_empty(line_config=lc)
        for params, res in params_res:
            pltb = StandardPlottable.from_simulation_result(res,
                                                            params,
                                                            extract_data_and_time)
            acc_error = pltb.apply_numpy_to_y(np.mean)
            bond_dim = params.bond_dim
            avg_pltb.add_point(bond_dim, acc_error)
        avg_pltb.sort_by_x()
        out.append(avg_pltb)
    return tuple(out)

def plot_accerr_vs_bd(md_filter: MetadataFilter,
              directory_path: str
              ) -> None:
    """
    Plots the bond dimension vs. accumulated error for the TDVP simulation results.
    """
    doc_style = DocumentStyle.THESIS
    config_matplotlib_to_latex(doc_style)
    size = set_size(doc_style, subplots=(1,3))

    pltbs: dict[TTNStructure, tuple[StandardPlottable, StandardPlottable]] = {}
    for struct in [TTNStructure.MPS, TTNStructure.BINARY, TTNStructure.TSTAR]:
        md_filter.change_criterium("structure", struct.value)
        pltbs[struct] = load_accerr_vs_bd(md_filter, directory_path)

    fig, ax = plt.subplots(1, 3, figsize=(size[0], 1.25*size[1]))
    for i, (struct, (std_pltb1, std_pltb2)) in enumerate(pltbs.items()):
        std_pltb1.plot_on_axis(ax[i])
        std_pltb2.plot_on_axis(ax[i])
        ax[i].set_xlabel("Bond Dimension")
        ax[i].set_yscale("log")
    ax[0].set_ylabel("Acc. Err.")
    ax[2].legend()
    plt.tight_layout()

    save_path = os.path.join(directory_path, "plots")
    save_path = os.path.join(save_path, "bd_vs_accerr.pdf")
    plt.savefig(save_path, bbox_inches="tight", format="pdf")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <data_directory>")
        sys.exit(1)

    data_dir = sys.argv[1]
    structure_configs = {TTNStructure.MPS: (12, 5, 64, 5),      # structure: (sys_size, min_bd, max_bd, step_bd)
                         TTNStructure.FTPS: (3, 2, 8, 1),
                         TTNStructure.BINARY: (3, 2, 8, 1),
                         TTNStructure.TSTAR: (3, 2, 8, 1)}
    
    md_filter = MetadataFilter()
    md_filter.change_criterium("structure", [TTNStructure.MPS.value,
                                             TTNStructure.BINARY.value,
                                             TTNStructure.TSTAR.value])
    md_filter.change_criterium("system_size", [12,3])
    md_filter.change_criterium("ext_magn", 0.5)
    md_filter.change_criterium("time_step_size", 0.01)
    plot_err_t(md_filter, data_dir)
    plot_bd_vs_rt(md_filter, data_dir)
    plot_accerr_vs_bd(md_filter, data_dir)
    md_filter.change_criterium("time_step_size", [0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
    md_filter.change_criterium("bond_dim", [64,8])  # Max bond dimension for time step dependence
    plot_dt_vs_final_error(md_filter, data_dir)
