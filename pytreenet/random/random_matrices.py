import numpy as np

from ..util.util import crandn

def random_hermitian_matrix(size: int = 2) -> np.ndarray:
    """
    Creates a random hermitian matrix H^\\dagger = H

    Args:
        size (int, optional): Size of the matrix. Defaults to 2.

    Returns:
        np.ndarray: The hermitian matrix.
    """
    if size < 1:
        errstr = "The dimension must be positive!"
        raise ValueError(errstr)
    matrix = crandn((size,size))
    return 0.5 * (matrix + matrix.T.conj())