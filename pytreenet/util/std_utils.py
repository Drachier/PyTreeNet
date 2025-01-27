"""
Some useful tools that do not fit into any other category.
"""
from typing import Tuple, Any, Dict, List
from copy import deepcopy, copy
from collections import Counter

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, eigsh
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
        Compares two lists by their identity (memory address) The elements and their orders should match.

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

def permute_tuple(tup: Tuple, permutation: List[int]) -> Tuple:
    """
    Permute the elements of a tuple.

    Args:
        tup (Tuple): The original tuple.
        permutation (List[int]): The permutation of the elements.
    
    Returns:
        Tuple: The permuted tuple.
    """
    assert len(tup) == len(permutation)
    return tuple(tup[i] for i in permutation)

def fast_exp_action(exponent: np.ndarray,
                    vector: np.ndarray,
                    mode: str = "fastest") -> np.ndarray:
    """
    Perform the action of the exponentiation of a matrix on a vector.

    Different modes can be choosen to perform the action. The fastest mode
    is the default mode. The modes are:

    - "expm": Use the scipy expm function.
    - "eigsh": Use the scipy eigsh function. Only valid for hermitian matrices.
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
    if mode == "eigsh":
        if exponent.shape[0] < 4:
            return expm(exponent) @ vector
        k = min(exponent.shape[0]-2, 8)
        w, v, = eigsh(exponent, k=k)
        return v @ np.diag(np.exp(w)) @ np.linalg.pinv(v) @ vector
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
