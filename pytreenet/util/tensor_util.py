"""
Helpful functions that work with the tensors of the tensor nodes.

These functions transform a given tensor independent of the overall tensor
network structure.
"""
from typing import Tuple, List, Union

from math import prod
import numpy as np

def transpose_tensor_by_leg_list(tensor: np.ndarray,
                                 first_legs: List[int],
                                 last_legs: List[int]) -> np.ndarray:
    """
    Transposes a tensor according to two lists of legs.
    
    The legs in the first_legs list will become the first legs of the new
    tensor and the legs in the secon dlist will become the last legs of the
    new tensor.

    Args:
        tensor (np.ndarray): Tensor to be transposed.
        first_legs (List[int]): Leg indices that are to become the first legs
            of the new tensor.
        last_legs (List[int]): Leg indices that are to become the last legs
            of the new tensor.
    
    Returns:
        np.ndarray: New tensor that is the transposed input tensor::

            _____       (0,2)       _____
        ___|     |___   (1,3)   ___|     |___
        0  |     |  1   ---->   0  |     |  2 
        ___|     |___           ___|     |___
        2  |_____|  3           1  |_____|  3

    """
    assert tensor.ndim == len(first_legs) + len(last_legs)
    correct_leg_order = first_legs + last_legs
    transposed_tensor = np.transpose(tensor, axes=correct_leg_order)
    return transposed_tensor

def tensor_matricization(tensor: np.ndarray,
                         output_legs: Tuple[int, ...],
                         input_legs: Tuple[int, ...],
                         correctly_ordered: bool = False) -> np.ndarray:
    """
    Turns a tensor into a matrix.
    
    This is done by combining the legs defined as output into one big output
    leg and the legs defined as input into one big input leg. The order of the
    legs is kept, i.e. the dimensions when viewed as a tensor product have the
    same order as the legs defined in the lists.

    Args:
        tensor (np.ndarray): Tensor to be matricized.
        output_legs (Tuple[int]): The tensor legs which are to be combined to
            be the matrix' output leg (First index).
        input_legs (Tuple[int]): The tensor legs which are to be combined to
            be the matrix' input leg (Second index).
        correctly_ordered (bool, optional): If true it is assumed, the tensor
            does not need to be transposed, i.e. this should be activated if the
            tensor already has the correct order of legs. Defaults to False.
    
    Returns:
        np.ndarray: The resulting matrix.
    """
    assert tensor.ndim == len(output_legs) + len(input_legs)
    if correctly_ordered:
        tensor_correctly_ordered = tensor
    else:
        tensor_correctly_ordered = transpose_tensor_by_leg_list(tensor,
                                                                output_legs,
                                                                input_legs)
    shape = tensor_correctly_ordered.shape
    output_dimension = prod(shape[0:len(output_legs)])
    input_dimension = prod(shape[len(output_legs):])
    matrix = np.reshape(tensor_correctly_ordered,
                        (output_dimension, input_dimension))
    return matrix

def tensor_matricisation_half(tensor: np.ndarray) -> np.ndarray:
    """
    Turns a tensor into a matrix combining half the legs into input atn output.

    More specifically the tensor is matricized by combining the first half of
    the legs to the input leg of the matrix and the second half of the legs to
    the output leg.

    Args:
        tensor (np.ndarray): Tensor to be matricized.
    
    Returns:
        np.ndarray: The resulting matrix.
    """
    assert tensor.ndim % 2 == 0
    output_legs = tuple(range(0,tensor.ndim//2))
    input_legs = tuple(range(tensor.ndim//2,tensor.ndim))
    return tensor_matricization(tensor, output_legs, input_legs,
                                correctly_ordered=True)

def compute_transfer_tensor(tensor: np.ndarray,
                            contr_indices: Union[Tuple[int,...], int]) -> np.ndarray:
    """
    Computes the tranfer tensor of the given tensor.
     
    The transfer tensor is the tensor that is obtained by contracting the
    tensor with its conjugate along the given indices.

    Args:
        tensor (np.ndarray): The tensor to compute the transfer tensor for.
        contr_indices (Union[Tuple[int], int]): The indices of the legs of the
            tensor to be contracted.

    Returns:
        np.ndarray: The transfer tensor, i.e.::

                 ____          ____
                |    |__oi1___|    |
             ___| A  |__oi2___| A* |____
                |____|        |____|
    
    """
    if isinstance(contr_indices, int):
        contr_indices = (contr_indices, )
    conj_tensor = np.conjugate(tensor)
    transfer_tensor = np.tensordot(tensor, conj_tensor,
                                   axes=(contr_indices, contr_indices))
    return transfer_tensor

def tensor_multidot(tensor: np.ndarray,
                    other_tensors: List[np.ndarray],
                    main_legs: List[int],
                    other_legs: List[int]) -> np.ndarray:
    """
    For a given tensor, perform multiple tensor contractions at once.

    The tensor is contracted with multiple other tensors. The legs for each
    for each contraction should be at the same position as the corresponding
    tensor in the list of tensors.

    Args:
        tensor (np.ndarray): Tensor to be mutliplied with other tensors.
        other_tensors (List[np.ndarray]): The tensors that should be
            contracted with tensor.
        main_legs (List[int]): The legs of tensor which are connected to the
            tensors in other_tensors.
        other_legs (List[int]): The legs of the tensors in other_tensors which
            are connected to tensor.

    Returns:
        np.ndarray: The resulting tensor
    """
    idx = np.argsort(main_legs)
    main_legs = [main_legs[i] for i in idx]
    other_tensors = [other_tensors[i] for i in idx]
    other_legs = [other_legs[i] for i in idx]
    connected_legs = 0
    for i, t in enumerate(other_tensors):
        tensor = np.tensordot(tensor, t, axes=(main_legs[i]-connected_legs, other_legs[i]))
        connected_legs += 1
    return tensor

def multi_kron(*args) -> np.ndarray:
    """
    Computes the kronecker product of multiple matrices.

    Args:
        *args: Matrices to be kroneckered.
    
    Returns:
        np.ndarray: The resulting matrix.
            The convention is that the righmost matrix, corresponds to the
            rightmost matrix in a usual kronecker product.

    """
    if len(args) == 1:
        return args[0]
    return np.kron(args[0], multi_kron(*args[1:]))

def make_last_leg_first(tensor: np.ndarray) -> np.ndarray:
    """
    Flips the last leg of a tensor to be the first leg.

    Args:
        tensor (np.ndarray): Tensor to be flipped.

    Returns:
        np.ndarray: The flipped tensor.
    """
    if tensor.ndim == 1:
        return tensor
    return np.moveaxis(tensor, -1, 0)
