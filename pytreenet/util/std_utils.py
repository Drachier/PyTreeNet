"""
Some useful tools that do not fit into any other category.
"""
from typing import Iterator, Any, Dict, List
from copy import deepcopy, copy
from collections import Counter

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.linalg import expm as expm_sparse
from scipy.sparse import csr_matrix

def copy_object(obj: Any, deep=True) -> Any:
    """
    Copy an object.

    Shortens the commonly used if-else statement for copying objects to
    distinguish between deep and shallow copies.

    Args:
        obj (Any): The object to copy.
        deep (bool, optional): Whether to perform a deep copy. Defaults to
          True.
    
    Returns:
        Any: The copied object.
    """
    if deep:
        new_obj = deepcopy(obj)
    else:
        new_obj = copy(obj)
    return new_obj

def sort_dictionary(dictionary: Dict) -> Dict:
    """
    Sort a dictionary by its values.

    Args:
        dictionary (Dict): The dictionary to sort.

    Returns:
        Dict: The sorted dictionary.
    """
    return dict(sorted(dictionary.items(),key = lambda ele: ele[1], reverse = False))

def compare_lists_by_value(list1: List, list2: List) -> bool:
    """
    Compare, if two lists have the same elements.

    Args:
        list1 (List): The first list.
        list2 (List): The second list.
    
    Returns:
        bool: Whether the two lists have the same elements.
    """
    if len(list1) != len(list2):
        return False
    if Counter(list1) == Counter(list2):
        return True
    return False

def compare_lists_by_identity(list1: List, list2: List) -> bool:
    """
    Compares two lists by their identity (memory address) The elements and
     their orders should match.

    Args:
        list1 (List): First list
        list2 (List): Second list

    Returns:
        bool: Whether the two lists have the same elements in the same order.
    """
    # Check if the lengths are the same
    if len(list1) != len(list2):
        return False
    # Compare the identity (memory address) of each object in the lists
    return all(id(obj1) == id(obj2) for obj1, obj2 in zip(list1, list2))

def permute_iterator(it: Iterator, permutation: List[int]) -> Iterator:
    """
    Permute the elements of a tuple.

    Args:
        tup (Tuple): The original tuple.
        permutation (List[int]): The permutation of the elements.
    
    Returns:
        Tuple: The permuted tuple.
    """
    assert len(it) == len(permutation)
    return it.__class__(it[i] for i in permutation)

def find_permutation(list1: List, list2: List) -> List[int]:
    """
    Find the permutation of the elements of list1 to match list2.

    Args:
        list1 (List): The first list.
        list2 (List): The second list.
    
    Returns:
        List[int]: The permutation of the elements of list1 to match list2,
            i.e. list2[i] = list1[permutation[i]] for all i.
    """
    assert len(list1) == len(list2)
    return [list2.index(x) for x in list1]

def is_broadcastable(shp1: Iterator, shp2: Iterator) -> bool:
    """
    Check if two shapes are broadcastable.
    """
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True

def int_to_slice(index: int) -> slice:
    """
    Convert an integer to a slice object.

    Args:
        index (int): The index to convert.

    Returns:
        slice: The slice object.
    """
    return slice(index, index+1)

def fast_exp_action(exponent: np.ndarray,
                    vector: np.ndarray,
                    mode: str = "fastest") -> np.ndarray:
    """
    Perform the action of the exponentiation of a matrix on a vector.

    Different modes can be choosen to perform the action. The fastest mode
    is the default mode. The modes are:

    - "expm": Use the scipy expm function.
    - "chebyshev": Use the scipy expm_multiply function.
    - "sparse": Use the scipy sparse expm function. Only valid for sparse
        matrices.
    - "none": Do not perform any action.

    Args:
        exponent (np.ndarray): The exponent in matrix form.
        vector (np.ndarray): The input vector in vector form.
        mode (str, optional): The mode to use. Defaults to "fastest".

    Raises:
        NotImplementedError: If an unimplemented mode is used.

    Returns:
        np.ndarray: The result of the exponentiation and multiplication.
          exp(exponent) @ vector.
    """
    if mode == "fastest":
        mode = "chebyshev"
    if mode == "expm":
        return expm(exponent) @ vector
    if mode == "chebyshev":
        return expm_multiply(exponent, vector,
                             traceA=np.trace(exponent))
    if mode == "sparse":
        exponent = csr_matrix(exponent)
        vector = csr_matrix(vector).transpose()
        exponent_ = expm_sparse(exponent)
        result = exponent_.dot(vector)
        return result.toarray()
    if mode == "none":
        return vector
    errstr = mode + " is not a possible mode for exponent action!"
    raise NotImplementedError(errstr)

def postivise_range(rng: range, size: int) -> range:
    """
    Postivise a range object. i.e. convert negative indices to
    positive indices with respect to the given size.

    Args:
        rng (range): The range object to postivise.
        size (int): The size with respect to which the range should be
          postivised. This is usually the number of legs of a node.

    Returns:
        range: The postivised range object.
    """
    if rng.start < 0:
        start = size + rng.start
    else:
        start = rng.start
    if rng.stop < 0:
        stop = size + rng.stop
    else:
        stop = rng.stop
    return range(start, stop, rng.step)

def average_data(data: List[np.ndarray]) -> np.ndarray:
    """
    Computes the average of a list of arrays.

    Args:
        data (List[np.ndarray]): The list of arrays to average.

    Returns:
        np.ndarray: The average of the input arrays.
    """
    if not data:
        raise ValueError("No data provided for averaging!")
    return np.mean(np.array(data), axis=0)

def identity_mapping(x: Any) -> Any:
    """
    Identity mapping.

    Args:
        x (Any): The input.

    Returns:
        Any: The output, which is the same as the input.
    """
    return x
