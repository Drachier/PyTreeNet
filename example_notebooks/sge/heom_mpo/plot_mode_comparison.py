import os

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pytreenet.ttno.state_diagram import TTNOFinder

def mode_to_save_str(mode):
    if mode == TTNOFinder.BIPARTITE:
        return "bipartite"
    elif mode == TTNOFinder.SGE:
        return "sge"
    else:
        raise ValueError("Invalid mode")

def homogenous_to_save_str(homogenous):
    if homogenous:
        return "homogenous"
    else:
        return "heterogenous"

def create_datapath(struture: str,
                    homogenous: bool,
                    mode: TTNOFinder
                    ) -> str:
    mode_str = mode_to_save_str(mode)
    homogenous_str = homogenous_to_save_str(homogenous)
    savepath = os.getcwd() + f"/data/data_{struture}_{mode_str}_{homogenous_str}.h5"
    return savepath

def _load_metadata(file):
    min_mol = file.attrs["min_mol"]
    max_mol = file.attrs["max_mol"]
    num_baths_min = file.attrs["num_baths_min"]
    num_baths_max = file.attrs["num_baths_max"]
    mol_nums = range(min_mol, max_mol+1)
    bath_nums = range(num_baths_min, num_baths_max+1)
    return mol_nums, bath_nums

def _plot_bond_dims(axs, datapath: str, data_type: str, **kwargs):
    with h5py.File(datapath, "r") as file:
        # Metadata
        mol_nums, _ = _load_metadata(file)
        num_bath = 10
        # Plotting
        bond_dims = np.array(file[data_type][str(num_bath)])
        axs.grid(True)
        axs.plot(mol_nums, bond_dims, **kwargs)

def _plot_avg_bond_dims(axs, datapath: str, **kwargs):
    _plot_bond_dims(axs, datapath, "avg_bond_dims", **kwargs)

def _plot_max_bond_dims(axs, datapath: str, **kwargs):
    _plot_bond_dims(axs, datapath, "max_bond_dims", **kwargs)

def _datatype_to_plotfct(datatype: str) -> callable:
    if datatype == "avg_bond_dims":
        return _plot_avg_bond_dims
    elif datatype == "max_bond_dims":
        return _plot_max_bond_dims
    else:
        raise ValueError("Invalid datatype!")

def create_savepath(homogenous: bool,
                    datatype: str
                    ) -> str:
    homogenous_str = homogenous_to_save_str(homogenous)
    savepath = os.getcwd() + f"/plots/bond_dim_{homogenous_str}_comp_{datatype}.pdf"
    return savepath

def mode_to_label_str(mode) -> str:
    if mode == TTNOFinder.BIPARTITE:
        return "Bipartite"
    elif mode == TTNOFinder.SGE:
        return "SGE"
    else:
        raise ValueError("Invalid mode")

def datatype_to_axislabel(datatype:str) -> str:
    if datatype == "avg_bond_dims":
        return "Average Bond Dimension"
    elif datatype == "max_bond_dims":
        return "Max Bond Dimension"
    else:
        raise ValueError("Invalid datatype!")

def combi_to_marker(structure: str,
                    mode: TTNOFinder) -> str:
    combi = (structure, mode)
    if combi == ("mps", TTNOFinder.BIPARTITE):
        return "1"
    if combi == ("mps", TTNOFinder.SGE):
        return "2"
    if combi == ("tree", TTNOFinder.BIPARTITE):
        return "3"
    if combi == ("tree", TTNOFinder.SGE):
        return "4"
    raise ValueError("Invalid combination")

def combi_to_linestyle(structure: str,
                     mode: TTNOFinder) -> str:
    combi = (structure, mode)
    if combi == ("mps", TTNOFinder.BIPARTITE):
        return "-"
    if combi == ("mps", TTNOFinder.SGE):
        return "--"
    if combi == ("tree", TTNOFinder.BIPARTITE):
        return "-"
    if combi == ("tree", TTNOFinder.SGE):
        return "--"
    raise ValueError("Invalid combination")

def combi_to_color(structure: str,
                     mode: TTNOFinder) -> str:
    combi = (structure, mode)
    if combi == ("mps", TTNOFinder.BIPARTITE):
        return "blue"
    if combi == ("mps", TTNOFinder.SGE):
        return "red"
    if combi == ("tree", TTNOFinder.BIPARTITE):
        return "orange"
    if combi == ("tree", TTNOFinder.SGE):
        return "green"
    raise ValueError("Invalid combination")


def main(homogenous: bool = True,
         datatype: str = "avg_bond_dims",
         save: bool = True):
    _, axs = plt.subplots(1, 1, figsize=(8,6))
    for mode in [TTNOFinder.BIPARTITE, TTNOFinder.SGE]:
        datapath = create_datapath("mps", homogenous, mode)
        markersize = 10
        _datatype_to_plotfct(datatype)(axs, datapath,
                                       label=f"MPS -{mode_to_label_str(mode)}",
                                       linestyle=combi_to_linestyle("mps", mode),
                                       marker=combi_to_marker("mps", mode),
                                       markersize=markersize,
                                       color=combi_to_color("mps", mode))
        datapath = create_datapath("tree", homogenous, mode)
        _datatype_to_plotfct(datatype)(axs, datapath,
                            label=f"Tree -{mode_to_label_str(mode)}",
                            linestyle=combi_to_linestyle("tree", mode),
                            marker=combi_to_marker("tree", mode),
                            markersize=markersize,
                            color=combi_to_color("tree", mode))
    axs.set_xlabel("$N$ (Number of Molecules)", fontsize=18)
    axs.set_ylabel(datatype_to_axislabel(datatype), fontsize=18)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if save:
        savepath = create_savepath(homogenous,datatype)
        plt.savefig(savepath, format="pdf")
    else:
        plt.show()

if __name__ == "__main__":
    for hom in [True,False]:
        for datatype in ["avg_bond_dims", "max_bond_dims"]:
            main(hom,datatype=datatype)
