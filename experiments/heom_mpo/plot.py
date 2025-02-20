import os

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pytreenet.ttno.state_diagram import TTNOFinder

def _load_metadata(file):
    min_mol = file.attrs["min_mol"]
    max_mol = file.attrs["max_mol"]
    num_baths_min = file.attrs["num_baths_min"]
    num_baths_max = file.attrs["num_baths_max"]
    mol_nums = range(min_mol, max_mol+1)
    bath_nums = range(num_baths_min, num_baths_max+1)
    return mol_nums, bath_nums

def _plot_bond_dims(axs, datapath: str, style: str = "solid",
                    label: str = ""):
    with h5py.File(datapath, "r") as file:
        # Metadata
        mol_nums, bath_nums = _load_metadata(file)
        # Plotting
        for num_bath in bath_nums:
            avg_bond_dims = np.array(file["avg_bond_dims"][str(num_bath)])
            max_bond_dims = np.array(file["max_bond_dims"][str(num_bath)])
            axs[0].plot(mol_nums, avg_bond_dims, linestyle=style)
            axs[1].plot(mol_nums, max_bond_dims, linestyle=style)
        axs[0].plot([], [], linestyle=style, label=label, c="black")
        axs[1].plot([], [], linestyle=style, label=label, c="black")

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

def _plot_mps(axs,
              homogenous: bool = True,
              mode: TTNOFinder = TTNOFinder.SGE):
    datapath = create_datapath("mps", homogenous, mode)
    _plot_bond_dims(axs, datapath, "--", "MPS")

def _plot_tree(axs,
               homogenous: bool = True,
               mode: TTNOFinder = TTNOFinder.SGE):
    datapath = create_datapath("tree", homogenous, mode)
    _plot_bond_dims(axs, datapath, ":", "Tree")

def create_savepath(homogenous: bool,
                    mode: TTNOFinder) -> str:
    mode_str = mode_to_save_str(mode)
    homogenous_str = homogenous_to_save_str(homogenous)
    savepath = os.getcwd() + f"/plots/bond_dim_{mode_str}_{homogenous_str}.pdf"
    return savepath

def main(homogenous: bool = True,
         mode: TTNOFinder = TTNOFinder.SGE,
         save: bool = True):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    _plot_mps(axs, homogenous, mode)
    _plot_tree(axs, homogenous, mode)
    axs[0].set_title("Average Bond Dimension")
    axs[0].set_xlabel("Number of Molecules")
    axs[0].set_ylabel("Average Bond Dimension")
    axs[1].set_title("Max Bond Dimension")
    axs[1].set_xlabel("Number of Molecules")
    axs[1].set_ylabel("Max Bond Dimension")
    plt.legend()
    if save:
        savepath = create_savepath(homogenous, mode)
        plt.savefig(savepath, format="pdf")
    else:
        plt.show()

if __name__ == "__main__":
    main()
