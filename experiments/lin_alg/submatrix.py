"""
An experiment to see how impactful a truncation of decomposition of a postive
semidefinite matrix is.
"""
from __future__ import annotations
from argparse import ArgumentParser

import numpy as np
from numpy.linalg import eigh
from h5py import File
import matplotlib.pyplot as plt

from pytreenet.util.crandn import (crandn,
                                   RandomDistribution)

class SingularResult:
    """
    A class to hold the results of a single run of the experiment.
    """
    def __init__(self, truncation_index: int,
                 normalized_error: np.float_,
                 full_truncation_error: np.float_):
        self.truncation_index = truncation_index
        self.normalized_error = normalized_error
        self.full_truncation_error = full_truncation_error

def one_run(size: int,
            seed: int | None = None,
            distribution: RandomDistribution = RandomDistribution.NORMAL,
            **kwargs
            ) -> list[SingularResult]:
    """
    Performs one run of the experiment.

    Args:
        size (int): The size of the positive semidefinite matrix.
        seed (int | None, optional): Seed for the random number generator.
            Defaults to None.
        distribution (RandomDistribution, optional): The distribution to
            sample from. Defaults to RandomDistribution.NORMAL.
        **kwargs: Additional keyword arguments for the random number
            generation.

    Returns:
        list[SingularResult]: The array of SingularResult objects.
    """
    outer_matrix = crandn((size, size),
                          seed=seed,
                          distribution=distribution,
                          **kwargs)
    inner_matrix = crandn((size, size),
                            seed=seed,
                            distribution=distribution,
                            **kwargs)
    inner_total = inner_matrix.T.conj() @ inner_matrix
    full_matrix = outer_matrix.T.conj() @ inner_total @ outer_matrix
    eig_vals, eig_vecs = eigh(inner_total)
    eig_vals_full, eig_vecs_full = eigh(full_matrix)
    full_norm = np.linalg.norm(full_matrix)
    out = []
    for i in range(1, size):
        # Truncating the decomposition of the matrix
        truncated_evals = eig_vals[i:]
        truncated_evecs = eig_vecs[:, i:]
        # Reconstructing the matrix from the truncated decomposition
        truncated_inner_matrix = truncated_evecs @ np.diag(truncated_evals) @ truncated_evecs.T.conj()
        difference = full_matrix - outer_matrix.T.conj() @ truncated_inner_matrix @ outer_matrix
        normed_error = np.linalg.norm(difference) / full_norm
        # Truncation of the full matrix
        truncated_evals_full = eig_vals_full[i:]
        truncated_evecs_full = eig_vecs_full[:, i:]
        full_truncated = truncated_evecs_full @ np.diag(truncated_evals_full) @ truncated_evecs_full.T.conj()
        full_truncation_error = np.linalg.norm(full_matrix - full_truncated) / full_norm
        out.append(SingularResult(i, normed_error, full_truncation_error))
    return out

def run_for_one_size(size: int,
                     seeds: list[int] | None = None,
                     **kwargs
                     ) -> dict[int, list[SingularResult]]:
    """
    Runs the experiment for a given size and multiple seeds.

    Args:
        size (int): The size of the positive semidefinite matrix.
        seeds (list[int] | None, optional): List of seeds for the random    
        number generator. If None, uses a default range of seeds.
            Defaults to None.
        **kwargs: Additional keyword arguments for the random number
            generation.

    Returns:
        dict[int, list[SingularResult]]: A dictionary mapping each seed
            to the corresponding list of SingularResult objects.
    """
    if seeds is None:
        keys = list(range(len(seeds)))
        seeds = [None for _ in range(len(seeds))]
    else:
        keys = seeds
    return {key: one_run(size, seed=seed, **kwargs)
            for key, seed in zip(keys, seeds)}

def run_for_multiple_sizes(sizes: list[int],
                           seeds: list[int] | None = None,
                           **kwargs
                           ) -> dict[int, dict[int, list[SingularResult]]]:
    """
    Runs the experiment for multiple sizes and multiple seeds.

    Args:
        sizes (list[int]): List of sizes of the positive semidefinite matrix.
        seeds (list[int] | None, optional): List of seeds for the random
            number generator. If None, uses a default range of seeds.
            Defaults to None.
        **kwargs: Additional keyword arguments for the random number
            generation.
    
    Returns:
        dict[int, dict[int, list[SingularResult]]]: A dictionary mapping each
            size to another dictionary that maps each seed to the corresponding
            list of SingularResult objects.
    """
    return {size: run_for_one_size(size, seeds=seeds, **kwargs)
            for size in sizes}

def save_results(results: dict[int, dict[int, list[SingularResult]]],
                 filename: str) -> None:
    """
    Saves the results of the experiment to a file.

    Args:
        results (dict[int, dict[int, list[SingularResult]]]): The results
            of the experiment.
        filename (str): The name of the file to save the results to.
    """
    with File(filename, 'w') as f:
        for size, seed_results in results.items():
            size_group = f.create_group(str(size))
            for seed, data in seed_results.items():
                seed_group = size_group.create_group(str(seed))
                truncation_indices = [result.truncation_index for result in data]
                full_truncation_errors = [result.full_truncation_error for result in data]
                errors = [result.normalized_error for result in data]
                seed_group.create_dataset('truncation_indices', data=truncation_indices)
                seed_group.create_dataset('full_truncation_errors', data=full_truncation_errors)
                seed_group.create_dataset('errors', data=errors)

def load_results(filename: str) -> dict[int, dict[int, list[SingularResult]]]:
    """
    Loads the results of the experiment from a file.

    Args:
        filename (str): The name of the file to load the results from.

    Returns:
        dict[int, dict[int, list[SingularResult]]]: The results of the
            experiment.
    """
    results = {}
    with File(filename, 'r') as f:
        for size in f.keys():
            size_group = f[size]
            seed_results = {}
            for seed in size_group.keys():
                seed_group = size_group[seed]
                truncation_indices = seed_group['truncation_indices'][:]
                errors = seed_group['errors'][:]
                full_truncation_errors = seed_group['full_truncation_errors'][:]
                seed_results[int(seed)] = [SingularResult(truncation_index, error, full_truncation_error)
                                           for truncation_index, error, full_truncation_error in zip(truncation_indices,
                                                                                                     errors,
                                                                                                     full_truncation_errors)]
            results[int(size)] = seed_results
    return results

def avrage_errors_over_seeds(results: dict[int, dict[int, list[SingularResult]]]
                             ) -> dict[int, list[SingularResult]]:
    """
    Averages the errors over the seeds for each size.

    Args:
        results (dict[int, dict[int, list[SingularResult]]]): The results
            of the experiment.
    
    Returns:
        dict[int, list[SingularResult]]: A dictionary mapping each size
            to the corresponding list of SingularResult objects containing the
            truncation index and the averaged normalized error.
    """
    averaged_results = {}
    for size, seed_results in results.items():
        # dict truncation index to list of errors
        error_per_truncation: dict[int, list[np.float_]] = {}
        full_error_per_truncation: dict[int, list[np.float_]] = {}
        for _, data in seed_results.items():
            for result in data:
                truncation_index = result.truncation_index
                error = result.normalized_error
                full_truncation_error = result.full_truncation_error
                if truncation_index not in error_per_truncation:
                    error_per_truncation[truncation_index] = []
                error_per_truncation[truncation_index].append(error)
                if truncation_index not in full_error_per_truncation:
                    full_error_per_truncation[truncation_index] = []
                full_error_per_truncation[truncation_index].append(full_truncation_error)
        averaged_results[size] = [SingularResult(truncation_index,
                                                 np.mean(errors),
                                                 np.mean(full_errors))
                                  for truncation_index, errors, full_errors in zip(error_per_truncation.keys(),
                                                                                   error_per_truncation.values(),
                                                                                   full_error_per_truncation.values())]
    return averaged_results

def plot_results(results: dict[int, dict[int, list[SingularResult]]],
                 filename: str) -> None:
    """
    Plots the results of the experiment.

    Args:
        results (dict[int, dict[int, list[SingularResult]]]): The results
            of the experiment.
        filename (str): The name of the file to save the plot to.
    """
    averaged_results = avrage_errors_over_seeds(results)
    for size, data in averaged_results.items():
        truncation_indices = [result.truncation_index for result in data]
        errors = [result.normalized_error for result in data]
        full_errors = [result.full_truncation_error for result in data]
        plt.plot(truncation_indices, errors, label=f"Size {size}")
        plt.plot(truncation_indices, full_errors, label=f"Size {size} (full truncation)",
                 linestyle='--')
    plt.yscale("log")
    plt.xlabel("Truncation Index")
    plt.ylabel("Normalized Error")
    plt.legend()
    plt.savefig(filename, format='pdf')
    plt.show()

def main():
    """
    Main function to run the experiment and save the results.
    
    The filename for saving the results is taken from the command line
    arguments.
    Via booleans in the command line arguments, the user can specify whether to run the
    experiment and/or plot the results. Both default to true.
    """
    parser = ArgumentParser(description="Run the submatrix experiment.")
    parser.add_argument('--filename', type=str, required=True,
                        help="The name of the file to save the results to.")
    parser.add_argument('--run', action='store_true', default=True,
                        help="Run the experiment and save the results.")
    parser.add_argument('--plot', action='store_true', default=True,
                        help="Plot the results from the saved file.")
    args = parser.parse_args()

    filename = args.filename
    if not filename.endswith('.h5'):
        filename += '.h5'
    if args.run:
        sizes = [100, 500, 1000, 5000]  # Example sizes
        seeds = [123554, 48312, 4839458, 23948, 2384820, 238384]
        results = run_for_multiple_sizes(sizes, seeds=seeds)
        save_results(results, filename)

    if args.plot:
        results = load_results(filename)
        plot_results(results, filename.replace('.h5', '.pdf'))

if __name__ == "__main__":
    main()
