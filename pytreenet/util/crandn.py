"""
Implements the random generation of complex arrays.
"""
from __future__ import annotations
from enum import Enum

import numpy as np
import numpy.random as npr

class RandomDistribution(Enum):
    """
    Enumeration of supported random distributions.
    """
    NORMAL = "normal"
    UNIFORM = "uniform"

    def generating_function(self, rng: npr.Generator):
        """
        Returns the random number generating function corresponding to the
        distribution.

        Args:
            rng (npr.Generator): The random number generator.

        Returns:
            Callable: The random number generating function.
        """
        if self == RandomDistribution.NORMAL:
            return rng.normal
        elif self == RandomDistribution.UNIFORM:
            return rng.uniform
        else:
            raise ValueError(f"Unknown distribution: {self}!")

def crandn(size: tuple[int, ...] | int,
           *args,
           seed=None,
           distribution: RandomDistribution = RandomDistribution.NORMAL,
           **kwargs
           ) -> np.ndarray:
    """
    Draw random samples from the standard complex normal (Gaussian)
      distribution.

    Args:
        size (tuple[int, ...] | int): The size/shape of the output array.
        *args: Additional dimensions (integers) to be added to the size.
        seed (int, optional): Seed for the random number generator.
            Defaults to None.
        distribution (RandomDistribution, optional): The distribution to
            sample from. Defaults to RandomDistribution.NORMAL.
        **kwargs: Additional keyword arguments for the random number
            generation.

    Returns:
        np.ndarray: The array of random complex numbers.

    Raises:
        ValueError: If *args contains non-integer values (e.g., tuples), as
            this would create ambiguity in the desired output shape.
    """
    # Validate that all args are integers (not tuples or other types)
    if len(args) > 0:
        for i, arg in enumerate(args):
            if not isinstance(arg, int):
                raise ValueError(
                    f"All additional dimensions (*args) must be integers. "
                    f"Got non-integer at position {i}: {type(arg).__name__}({arg}). "
                    f"If you want to specify a complete shape, pass it as a single tuple."
                )
    
    rng = npr.default_rng(seed)
    gen_func = distribution.generating_function(rng)
    if isinstance(size, int) and len(args) > 0:
        size = tuple([size] + list(args))
    elif isinstance(size,int):
        size = (size,)
    # 1/sqrt(2) is a normalization factor
    return (gen_func(size=size, **kwargs) +
            1j*gen_func(size=size, **kwargs)) / np.sqrt(2)

def crandn_like(array: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Draw random samples from the standard complex normal (Gaussian)
      distribution with the same shape as the input array.

    Args:
        array (np.ndarray): The input array.
        *args: Additional arguments for the random number generation.
        **kwargs: Additional keyword arguments for the random number
            generation.

    Returns:
        np.ndarray: The array of random complex numbers.
    """
    return crandn(array.shape, *args, **kwargs)
