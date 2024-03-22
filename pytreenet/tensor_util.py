"""
Helpfull functions that work with the tensors of the tensor nodes.
"""
from __future__ import annotations
from typing import Tuple, List, Union
from enum import Enum

from math import prod
import numpy as np

class SplitMode(Enum):
    FULL = "full"
    REDUCED = "reduced"
    KEEP = "keep"

    def numpy_qr_mode(self) -> str:
        """
        Returns the string required in the numpy QR decomposition.
        """
        if self is SplitMode.FULL:
            return "complete"
        return "reduced"

def transpose_tensor_by_leg_list(tensor: np.ndarray,
                                 first_legs: List[int],
                                 last_legs: List[int]) -> np.ndarray:
    """
    Transposes a tensor according to two lists of legs. All legs in first_legs
    will become the first legs of the new tensor and the last_legs will all
    become the last legs of the tensor.

    Args:
        tensor (np.ndarray): Tensor to be transposed.
        first_legs (List[int]): Leg indices that are to become the first legs
         of the new tensor.
        last_legs (List[int]): Leg indices that are to become the last legs
         of the new tensor.
    
    Returns:
        np.ndarray: New tensor that is the transposed input tensor.

    Example:
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
    Turns a tensor into a matrix by combining the output_legs to the output
    leg of the matrix and the input_legs to the input leg of the matrix.

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

def _determine_tensor_shape(old_shape: Tuple[int,...],
                            matrix: np.ndarray,
                            legs: Tuple[int,...],
                            output: bool = True) -> Tuple[int,...]:
    """
    Determines the new shape a matrix is to be reshaped to after a decomposition
     of a tensor with spape old_shape, to again obtain a tensor.
    Works only if all legs to be reshaped are combined in the input or output
     leg of the matrix.
    
    Args:
        old_shape (Tuple[int]): Shape of the original tensor.
        matrix (np.ndarray): Matrix to be reshaped.
        legs (Tuple[int]): Which legs of the original tensor are associated to
         the matrix.
        output (bool, optional): If the legs of the original tensor are
         associated to the input or output of matrix. Defaults to True.
    
    Returns:
        Tuple[int]: New shape to which matrix is to be reshaped.
    """
    leg_shape = [old_shape[i] for i in legs]
    if output:
        matrix_dimension = [matrix.shape[1]]
        leg_shape.extend(matrix_dimension)
        new_shape = leg_shape
    else:
        matrix_dimension = [matrix.shape[0]]
        matrix_dimension.extend(leg_shape)
        new_shape = matrix_dimension

    return tuple(new_shape)

def tensor_qr_decomposition(tensor: np.ndarray,
                            q_legs: Tuple[int,...],
                            r_legs: Tuple[int,...],
                            mode: SplitMode = SplitMode.REDUCED) -> Tuple[np.ndarray,np.ndarray]:
    """
    Computes the QR decomposition of a tensor with respect to the given legs.

    Args:
        tensor (np.ndarray): Tensor on which the QR-decomp is to applied.
        q_legs (Tuple[int]): Legs of tensor that should be associated to the
         Q tensor after QR-decomposition.
        r_legs (Tuple[int]): Legs of tensor that should be associated to the
         R tensor after QR-decomposition.
        mode (SplitMode, optional): Reduced returns a QR deocomposition with
         minimum dimension between Q and R. Full returns the decomposition
         with dimension between Q and R as the output dimension of Q. Keep
         causes Q to have the same shape as the input tensor. Defaults to
         SplitMode.Reduced.

    Returns:
        Tuple[np.ndarray,np.ndarray]: (Q, R)

    Example:
             |2                             |1
           __|_      r_legs = (1, )       __|_        ____
          |    |     q_legs = (0,2)      |    |      | R  |
       ___|    |___  -------------->  ___| Q  |______|____|____
       0  |____|  1                   0  |____| 2   0        1


    """
    correctly_order = q_legs + r_legs == list(range(len(q_legs) + len(r_legs)))
    matrix = tensor_matricization(tensor, q_legs, r_legs,
                                  correctly_ordered=correctly_order)
    q, r = np.linalg.qr(matrix, mode=mode.numpy_qr_mode())
    shape = tensor.shape
    q_shape = _determine_tensor_shape(shape, q, q_legs, output=True)
    r_shape = _determine_tensor_shape(shape, r, r_legs, output=False)
    q = np.reshape(q, q_shape)
    r = np.reshape(r, r_shape)
    if mode is SplitMode.KEEP:
        orig_bond_dim = np.prod(r.shape[1:])
        diff = orig_bond_dim - q.shape[-1]
        padding_q = [(0,0)] * (q.ndim-1)
        padding_q.append((0,diff))
        q = np.pad(q,padding_q)
        padding_r = [(0,diff)]
        padding_r.extend([(0,0)]*(r.ndim-1))
        r = np.pad(r,padding_r)
    return q, r

def tensor_svd(tensor: np.ndarray,
               u_legs: Tuple[int,...],
               v_legs: Tuple[int,...],
               mode: SplitMode = SplitMode.REDUCED) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Perform a singular value decomposition on a tensor.

    Args:
        tensor (np.ndarray): Tensor on which the svd is to be performed.
        u_legs (Tuple[int]): Legs of tensor that are to be associated to U
         after the SVD.
        v_legs (Tuple[int]): Legs of tensor that are to be associated to V
         after the SVD.
        mode (SplitMode, optional): Determines if the full or reduced matrices
         u and vh obtained by the SVD are returned. The default is
         SplitMode.REDUCED.
    
    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray]: (U, S, V)

    Example:
             |2                             |1
           __|_      v_legs = (1, )       __|_        ____        ____
          |    |     q_legs = (0,2)      |    |      |  S |      |    |
       ___|    |___  -------------->  ___|  U |______|____|______| Vh |____1 
       0  |____|  1                   0  |____| 2   0       1  0 |____|
    """
    # TODO: Modify to use the Enum
    if (mode != "reduced") and (mode != "full"):
        raise ValueError(f"'mode' may only be 'full' or 'reduced' not {mode}!")
    if u_legs + v_legs == list(range(len(u_legs) + len(v_legs))):
        correctly_order = True
    else:
        correctly_order = False
    # Cases to deal with input format of numpy function
    if mode == 'full':
        full_matrices = True
    elif mode == 'reduced':
        full_matrices = False

    matrix = tensor_matricization(tensor, u_legs, v_legs, correctly_ordered=correctly_order)
    u, s, vh = np.linalg.svd(matrix, full_matrices=full_matrices)
    shape = tensor.shape
    u_shape = _determine_tensor_shape(shape, u, u_legs, output=True)
    vh_shape = _determine_tensor_shape(shape, vh, v_legs, output=False)
    u = np.reshape(u, u_shape)
    vh = np.reshape(vh, vh_shape)

    return u, s, vh

def check_truncation_parameters(max_bond_dim: int,
                                rel_tol: float,
                                total_tol: float):
    """
    Checks if the truncation parameters are valid.

    Parameters:
        max_bond_dim (int): The maximum bond dimension allowed between nodes.
        rel_tol (float): singular values s for which ( s / largest singular value)
         < rel_tol are truncated.
        total_tol (float): singular values s for which s < total_tol are truncated.
    """
    if (not isinstance(max_bond_dim,int)) and (max_bond_dim != float("inf")):
        raise TypeError(f"'max_bond_dim' has to be int not {type(max_bond_dim)}!")
    if max_bond_dim < 0:
        raise ValueError("'max_bond_dim' has to be positive.")
    if (rel_tol < 0) and (rel_tol != float("-inf")):
        raise ValueError("'rel_tol' has to be positive or -inf.")
    if (total_tol < 0) and (total_tol != float("-inf")):
        raise ValueError("'total_tol' has to be positive or -inf.")

def truncated_tensor_svd(tensor: np.ndarray,
                         u_legs: Tuple[int,...],
                         v_legs: Tuple[int,...],
                         max_bond_dim: int = 100,
                         rel_tol: float = 0.01,
                         total_tol: float = 1e-15):
    """
    Performs a singular value decomposition of a tensor including truncation,
     i.e. discarding some singular values.
    
    Args:
        tensor (np.ndarray): Tensor on which the svd is to be performed.
        u_legs (Tuple[int]): Legs of tensor that are to be associated to U
         after the SVD.
        v_legs (Tuple[int]): Legs of tensor that are to be associated to V
         after the SVD.
        max_bond_dim (int, optional): The maximum bond dimension allowed
         between nodes. Defaults to 100.
        rel_tol (float, optional): singular values s for which
         (s / largest singular value) < rel_tol are truncated. Defaults to 0.01.
        total_tol (float, optional): singular values s for which s < total_tol
         are truncated. Defaults to 1e-15.
    
    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray]: (U, S, V)

    Example:
             |2                             |1
           __|_      v_legs = (1, )       __|_        ____        ____
          |    |     q_legs = (0,2)      |    |      |  S |      |    |
       ___|    |___  -------------->  ___|  U |______|____|______| Vh |____1 
       0  |____|  1                   0  |____| 2   0       1  0 |____|
    """
    check_truncation_parameters(max_bond_dim, rel_tol, total_tol)
    correctly_order =  u_legs + v_legs == list(range(len(u_legs) + len(v_legs)))

    matrix = tensor_matricization(tensor, u_legs, v_legs, correctly_ordered=correctly_order)
    u, s, vh = np.linalg.svd(matrix)

    # Here the truncation happens
    max_singular_value = s[0]
    min_singular_value_cutoff = max(rel_tol * max_singular_value, total_tol)
    s_temp = [singular_value for singular_value in s
         if singular_value > min_singular_value_cutoff]

    if len(s_temp) > max_bond_dim:
        s = s_temp[:max_bond_dim]
    elif len(s_temp) == 0:
        s = [max_singular_value]
    else:
        s = s_temp

    new_bond_dim = len(s)
    u = u[:, :new_bond_dim]
    vh = vh[:new_bond_dim, :]

    shape = tensor.shape
    u_shape = _determine_tensor_shape(shape, u, u_legs, output=True)
    vh_shape = _determine_tensor_shape(shape, vh, v_legs, output=False)
    u = np.reshape(u, u_shape)
    vh = np.reshape(vh, vh_shape)

    return u, np.asarray(s), vh


def contr_truncated_svd_splitting(tensor: np.ndarray,
                                  u_legs: Tuple[int,...],
                                  v_legs: Tuple[int,...],
                                  **truncation_param):
    """
    Performs a truncated singular value decomposition, but the singular values
     are contracted with the V tensor.

    Args:
        tensor (np.ndarray): Tensor on which the svd is to be performed.
        u_legs (Tuple[int]): Legs of tensor that are to be associated to U
         after the SVD.
        v_legs (Tuple[int]): Legs of tensor that are to be associated to V
         after the SVD.
        **truncation_param: Parameters for the truncation of the singular values.
         (See truncated_tensor_svd)
    """
    u, s, vh = truncated_tensor_svd(tensor, u_legs, v_legs, **truncation_param)
    svh = np.tensordot(np.diag(s), vh, axes=(1, 0))
    return u, svh

def compute_transfer_tensor(tensor: np.ndarray,
                            contr_indices: Union[Tuple[int,...], int]) -> np.ndarray:
    """
    Computes the tranfer tensor of the given tensor with respect to the
     indices given. This means it contracts the tensor with its conjugate
     along the given indices.

    Args:
        tensor (np.ndarray): The tensor to compute the transfer tensor for.
        contr_indices (Union[Tuple[int], int]): The indices of the legs of the
         tensor to be contracted

    Returns:
        np.ndarray: The transfer tensor, i.e.
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
