"""
This is a small script to show some application of the singular value
decomposition.
"""
import os
import pickle
import sys

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib

from pytreenet.random.random_matrices import crandn
from pytreenet.util.tensor_splitting import tensor_svd
from pytreenet.util.plotting import (config_matplotlib_to_latex,
                                     set_size)

def run_random(num_samples: int,
               dimension: int
               ) -> NDArray[np.complex64]:
    """
    Run a singular value decomposition on a number of random matrices
    and return the average singular values.

    Args:
        num_samples (int): The number of random matrices to generate.
        dimension (int): The dimension of the square matrices.
    
    Returns:
        NDArray[np.complex64]: The average singular values of the random
            matrices.
    """
    results = np.zeros((num_samples, dimension), dtype=float)
    for i in range(num_samples):
        random_matrix = crandn((dimension,dimension))
        _, s, _ = tensor_svd(random_matrix, (0, ), (1, ))
        s_norm = s / s[0]
        results[i, :] = s_norm.astype(np.float64)
    avg = np.mean(results, axis=0)
    return avg

def run_cifar_10(dir_path: str
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Run a singular value decomposition on batch 1 of the CIFAR-10 dataset.

    Args:
        dir_path (str): The path to the CIFAR-10 dataset directory.

    Returns:
        tuple[NDArray[np.float]]: The average singular values of the CIFAR-10.
    """
    filepath = os.path.join(dir_path, "data_batch_1")
    with open(filepath, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        data = batch[b"data"]
    side_length = 32
    num_samples = data.shape[0]
    num_pixels = side_length ** 2
    colour_map = {"red": 0, "green": 1, "blue": 2}
    colour_data = [data[:, num_pixels * colour:num_pixels * (colour + 1)]
                         for colour in colour_map.values()]
    colour_matrices = [colour.reshape((num_samples,side_length,side_length))
                       for colour in colour_data]
    sing_values = [np.linalg.svd(colour_matrix, compute_uv=False)
                   for colour_matrix in colour_matrices]
    # Normalize the singular values
    sing_values = [sing_value / sing_value[:,[0]] for sing_value in sing_values]
    avg_sing_values = [np.mean(sing_value, axis=0) for sing_value in sing_values]
    return tuple(avg_sing_values)

def main(dir_path: str,
         plot_path: str | None = None
         ) -> None:
    """
    Main function to run the SVD on CIFAR-10 and print the results.

    Args:
        dir_path (str): The path to the CIFAR-10 dataset directory.
        plot_path (str): The path to save the plot. If None, the plot will not
            be saved.
    """
    num_samples = 10000
    dimensions = 32
    # Run the random matrix SVD
    rand_avg = run_random(num_samples, dimensions)
    print(f"Average singular values of {num_samples} random {dimensions}x{dimensions} matrices:")
    print(rand_avg)
    # Run the CIFAR-10 SVD
    print(f"Running SVD on CIFAR-10 batch 1 from {dir_path}...")
    avg_sing_values = run_cifar_10(dir_path)
    print("Average singular values of CIFAR-10 batch 1:")
    print(f"Red: {avg_sing_values[0]},")
    print(f"Green: {avg_sing_values[1]},")
    print(f"Blue: {avg_sing_values[2]}")
    # Plot the results
    plot(rand_avg, run_cifar_10(dir_path), savepath=plot_path)

def plot(sing_values: NDArray[np.float64],
         cifar10_vals: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
         savepath: str | None = None
         ) -> None:
    """
    Plot the singular values of random matrices.

    Args:
        sing_values (NDArray[np.float64]): The singular values to plot.
        cifar10_vals (tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]):
            The singular values of the CIFAR-10 dataset.
        savepath (str | None): The path to save the plot. If None, the plot
            will not be saved.
    """
    matplotlib.use("Agg")
    config_matplotlib_to_latex()
    fig_width, fig_height = set_size("thesis")
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(sing_values,
             marker="s",
             ms=5,
             linestyle=" ",
             color="black",
             label="random")
    for i, design in enumerate([("red","o"), ("green","^"), ("blue","*")]):
        colour, marker = design
        plt.semilogy(cifar10_vals[i],
                 marker=marker,
                 alpha=0.7,
                 ms=5,
                 linestyle=" ",
                 label=f"CIFAR-10 {colour}",
                 color=colour)
    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value")
    plt.xlim(left=0)
    plt.grid(alpha=0.5)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", format="pdf")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python svd.py <path_to_cifar10_directory>")
        sys.exit(1)
    cifar10_dir = sys.argv[1]
    plot_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(cifar10_dir, plot_path=plot_path)
