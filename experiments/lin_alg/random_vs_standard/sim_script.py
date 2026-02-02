"""
Implements the simulation of standard SVD vs random SVD on matrices.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from time import time
import os

import numpy as np
import numpy.typing as npt
from h5py import File

from pytreenet.util.matrix_decomp.randomized_svd import (randomized_svd,
                                                         randomised_qb,
                                                         SubspaceIterationNumber)
from pytreenet.util.experiment_util.sim_params import SimulationParameters
from pytreenet.random.random_matrices import (crandn, RandomDistribution)
from pytreenet.time_evolution.results import Results
from pytreenet.util.experiment_util.script_util import script_main

class SVDType(Enum):
    """
    Enum for specifying the type of SVD to use in the simulation.

    Attributes:
        STANDARD (str): Use the standard SVD.
        RANDOMIZED (str): Use the randomized SVD.
        QB (str): Use the randomized QB decomposition.
    """
    STANDARD = "standard"
    RANDOMIZED = "randomized"
    QB = "qb"

@dataclass
class RandomVsStandardParams(SimulationParameters):
    """
    Parameters for the random vs standard SVD simulation.
    """
    svd_type: SVDType = SVDType.STANDARD
    seed: int = 168433857
    low: float = -0.5
    high: float = 1
    dimension: int = 100
    rank_min: int = 5
    rank_max: int = 100
    rank_step: int = 5

def generate_matrix(params: RandomVsStandardParams
                    ) -> npt.NDArray[np.complex64]:
    """
    Generate a random matrix based on the simulation parameters.

    Args:
        params (RandomVsStandardParams): The simulation parameters.

    Returns:
        np.ndarray: The generated random matrix.
    """
    matrix = crandn(size=(params.dimension, params.dimension),
                    distribution=RandomDistribution.UNIFORM,
                    seed = params.seed,
                    low=params.low,
                    high=params.high)
    return matrix

def truncated_svd(matrix: npt.NDArray[np.complex64],
                  rank: int,
                  params: RandomVsStandardParams
                  ) -> npt.NDArray[np.complex64]:
    """
    Perform truncated SVD on the given matrix based on the simulation parameters.

    Args:
        matrix (np.ndarray): The input matrix to decompose.
        rank (int): The rank for the truncated SVD.
        params (RandomVsStandardParams): The simulation parameters.

    Returns:
        npt.NDArray[np.complex64]: The matrix resulting from the truncated SVD.
            U @ S @ Vh where U, S, Vh are truncated to the specified rank.
    """
    if params.svd_type == SVDType.STANDARD:
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        return u[:, :rank] @ np.diag(s[:rank]) @ vh[:rank, :]
    if params.svd_type == SVDType.RANDOMIZED:
        u, s, vh =  randomized_svd(matrix,
                                   rank,
                                   seed=params.seed+1,
                                   n_subspace_iters=SubspaceIterationNumber.NONE)
        return u @ np.diag(s) @ vh
    if params.svd_type == SVDType.QB:
        q_mat, b_mat = randomised_qb(matrix,
                                     rank,
                                     seed=params.seed+2,
                                     n_subspace_iters=SubspaceIterationNumber.NONE)
        return q_mat @ b_mat
    errstr = "Invalid SVD type specified: " + params.svd_type.value + "!"
    raise ValueError(errstr)

def compute_svd_error(params: RandomVsStandardParams
                      ) -> Results:
    """
    Compute the error of the truncated SVD approximation.

    Args:
        params (RandomVsStandardParams): The simulation parameters.

    Returns:
        Results: The results of the SVD error computation.
    """
    matrix = generate_matrix(params)
    mat_norm = np.linalg.norm(matrix, ord='fro')
    rank_range = range(params.rank_min, params.rank_max + 1, params.rank_step)
    results = Results()
    results.initialize({"error": float,
                        "runtime": float,
                        "rank": int},
                        len(list(rank_range)),
                        with_time=False)
    for index, rank in enumerate(rank_range):
        start = time()
        approx_matrix = truncated_svd(matrix, rank, params)
        run_time = time() - start
        error = np.linalg.norm(matrix - approx_matrix, ord='fro')
        normed_error = error / mat_norm
        results.set_element("error",
                            index,
                            normed_error)
        results.set_element("runtime",
                            index,
                            run_time)
        results.set_element("rank",
                            index,
                            rank)
    return results

def run_and_save(params: RandomVsStandardParams,
                 save_path: str
                 ) -> None:
    """
    Run the SVD error computation and save the results.

    Args:
        params (RandomVsStandardParams): The simulation parameters.
        save_path (str): The path to save the results.
    """
    results = compute_svd_error(params)
    hash_val = params.get_hash()
    file_path = os.path.join(save_path, f"{hash_val}.h5")
    with File(file_path, "w") as f:
        results.save_to_h5(f)
        params.save_to_h5(f)

if __name__ == "__main__":
    script_main(run_and_save, RandomVsStandardParams)
