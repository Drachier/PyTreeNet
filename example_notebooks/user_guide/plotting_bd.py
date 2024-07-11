from typing import Union
import re
from argparse import ArgumentParser

import h5py
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm

FIGSIZE = (4,3.5)

def plot_bond_dimensions_for_max_terms(num_terms: int,
                                       file_path: str,
                                       save_path: Union[str,None]=None):
    """
    Plots the bond dimensions from a given file for a given maximum number of
    terms.

    The file is then saved to a different location as a pdf.
    """
    with h5py.File(file_path, "r") as file:
        svd_bonds = file[f"run_with_{num_terms}_terms"]["svd_bond_dim"][0:-1]
        ham_bonds = file[f"run_with_{num_terms}_terms"]["state_diag_bond_dim"][0:-1]

    paired_dims = list(zip(ham_bonds.flatten(), svd_bonds.flatten()))
    paired_set = set(paired_dims)

    paired_counts = np.asarray([[pair[0], pair[1], paired_dims.count(pair)] for pair in paired_set])
    fig2 = plt.figure(figsize=FIGSIZE)

    # As a reference
    plt.plot(range(1,num_terms + 1), range(1,num_terms + 1), lw=1, zorder=-1)

    plt.scatter(paired_counts[:,1], paired_counts[:,0], c=paired_counts[:,2], cmap='YlOrRd',
                norm=LogNorm(), s=5, zorder=10)
    plt.ylabel("Bond Dimensions via State Diagram")
    plt.xlabel("Bond Dimensions via SVD")
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches='tight')
    else:
        plt.show()

def input_handling():
    """
    Handles the input parsed from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str, nargs=1)
    args = vars(parser.parse_args())
    filepath = args["filepath"][0]
    if not re.match(r".+\.hdf5$", filepath):
        loadpath = filepath + ".hdf5"
        savepath = filepath + ".pdf"
    else:
        loadpath = filepath
        savepath = filepath[:-5] + ".pdf"
    print("Loading Data from " + filepath)
    print("Saving Plot to " + savepath)
    return loadpath, savepath

if __name__ == "__main__":
    load, save = input_handling()
    plot_bond_dimensions_for_max_terms(30, load, save)
