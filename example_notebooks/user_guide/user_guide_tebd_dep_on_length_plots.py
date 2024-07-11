
from typing import Tuple
import re

import h5py
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors, cm

from tebd_user_guide_util import input_handling_filepath, create_file_path

def get_filepaths() -> Tuple[str, str]:
    parser = input_handling_filepath()
    args = vars(parser.parse_args())
    filepath = args["filepath"][0]
    load_path = create_file_path(filepath)
    save_path = create_file_path(filepath, ".pdf")
    return load_path, save_path

def plot():
    load_path, save_path = get_filepaths()
    with h5py.File(load_path, "r") as file:
        figure, ax = plt.subplots(figsize=(7,5))
        times = file["times"][:]
        for data_id in file:
            if re.match(r"^length_\d+$", data_id):
                length = int(data_id.split("_")[1])
                magn = file[data_id]["magn"][:]
                if length % 5 == 0:
                    colour = colormaps.get_cmap("inferno")(length / 100)
                    ax.plot(times, magn, label=f"Length {length}", color=colour)
            else:
                print(f"Skipping {data_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\langle M \\rangle$")
    plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap='inferno'), ax=ax,
                 label=r"$L / L_{\max}$")
    plt.savefig(save_path, format="pdf", bbox_inches='tight')

if __name__ == "__main__":
    plot()
