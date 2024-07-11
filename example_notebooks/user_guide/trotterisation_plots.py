import re
from argparse import ArgumentParser

import h5py
import matplotlib.pyplot as plt

def create_marker_dict():
    return {"first_order": "+",
            "strang_split": "x"}

def colours():
    return {"mblue": (0.368417, 0.506779, 0.709798),
            "morange": (0.880722, 0.611041, 0.142051),
            "mgreen": (0.560181, 0.691569, 0.194885),
            "mred": (0.922526, 0.385626, 0.209179),
            "mpurple": (0.528488, 0.470624, 0.701351),
            "mbrown": (0.772079, 0.431554, 0.102387),
            "mpink": (0.910569, 0.506117, 0.755282),
            "mlightblue": (0.676383, 0.866289, 0.803225),}

def create_colour_dict():
    cs = colours()
    return {"00": cs["mblue"],
            "01": cs["morange"],
            "10": cs["mgreen"],
            "11": cs["mred"],
            "phi0": cs["mpurple"],
            "phi1": cs["mbrown"],
            "phi2": cs["mpink"],
            "phi3": cs["mlightblue"]}

def format_state_label(state_label: str) -> str:
    if re.match(r"phi\d", state_label):
        return r"$\left|\phi_{" + state_label[-1] + r"}\right>$"
    if re.match(r"\d{2}", state_label):
        return r"$\left|{" + state_label[0] + r"}{" + state_label[1] + r"}\right>$"

def input_handling():
    parser = ArgumentParser()
    parser.add_argument("filepath", nargs="+", type=str,
                        help="Filepath to the HDF5 file.")
    args = vars(parser.parse_args())
    filepath = args["filepath"][0]
    if re.match(r".+\.hdf5", filepath):
        return filepath, filepath[:-5] + ".pdf"
    else:
        return filepath + ".hdf5", filepath + ".pdf"

if __name__ == "__main__":
    filepath, savepath = input_handling()
    with h5py.File(filepath) as file:
        time_step_sizes = file["time_step_sizes"][:]
        results = {}
        for split_name in file["results"]:
            results[split_name] = {}
            for basis_name in file["results"][split_name]:
                results[split_name][basis_name] = {}
                for state_id in file["results"][split_name][basis_name]:
                    results[split_name][basis_name][state_id] = file["results"][split_name][basis_name][state_id][:]

    figure = plt.figure(figsize=(7,5))
    marker_dict = create_marker_dict()
    colour_dict = create_colour_dict()
    for i, split_name in enumerate(results):
        split_results = results[split_name]
        for basis_name, basis_results in split_results.items():
            for state_id, state_results in basis_results.items():
                plt.loglog(time_step_sizes, state_results,
                           marker=marker_dict[split_name], color=colour_dict[state_id],
                           linestyle="None",ms=5)
                if split_name == "first_order":
                    plt.plot([],[],label=format_state_label(state_id),
                             marker="s", color=colour_dict[state_id], linestyle="None", ms=5)
    # Plot dt=dt and dt=dt^2 lines
    plt.plot(time_step_sizes, [dt * 0.16 for dt in time_step_sizes], label=r"$\mathcal{O} (\Delta t)$",
             color="black")
    plt.plot(time_step_sizes, [dt**2 * 0.08 for dt in time_step_sizes], label=r"$\mathcal{O}(\Delta t^2)$",
             color="black", linestyle="--")
    plt.legend()
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error")
    plt.savefig(savepath, format="pdf", bbox_inches='tight')
