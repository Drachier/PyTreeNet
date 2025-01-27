"""
Functions that decompose a tensor.

The functions in this module are used to decompose a tensor into two or more
tensors. The QR decomposition and the singular value decomposition are
the most common decompositions used in tensor networks and implemented.
While the QR-Decomposition is faster, the SVD allows for a truncation of the
bond dimension, by discarding sufficiently small singular values.
"""
from typing import Tuple, Union
from enum import Enum
from warnings import warn
from dataclasses import dataclass

import numpy as np

from .tensor_util import tensor_matricization
from .ttn_exceptions import positivity_check

class SplitMode(Enum):
    """
    Different modes on how to deal with the dimensions of a decomposed tensor.

    Decompositions are not unique, but differ in how to deal with the dimension
    of the new created bond. In principle one can increase the dimension as
    much as is desired, by adding zero vectors to the resulting tensors.

    FULL: The resulting tensor has the keeps the inout dimension of the
        original tensor.
    REDUCED: The resulting tensor has the minimum dimension between the
        input and output tensor. This means there are no zero vectors added.
        This is the default mode of numpy.
    KEEP: One of the resulting tensors has the same shape as the input tensor,
        if only one leg is split of from the rest.
    """
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

    def numpy_svd_mode(self) -> bool:
        """
        Returns the numpy SVD mode required.

        A numpy SVD has to be told, if it should return the full or reduced
        matrices by using a boolean.
        """
        return self is not SplitMode.REDUCED

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
            SplitMode.REDUCED.

    Returns:
        Tuple[np.ndarray,np.ndarray]: (Q, R)::

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
        Tuple[np.ndarray,np.ndarray,np.ndarray]: (U, S, V), where S is the
            vector of singular values and U and V are tensors equivalent to an
            isometry::

                      |2                             |1
                    __|_      v_legs = (1, )       __|_        ____        ____
                   |    |     q_legs = (0,2)      |    |      |  S |      |    |
                ___|    |___  -------------->  ___|  U |______|____|______| Vh |____1 
                0  |____|  1                   0  |____| 2   0       1  0 |____|

    
    """
    correctly_ordered = u_legs + v_legs == list(range(len(u_legs) + len(v_legs)))
    full_matrices = mode.numpy_svd_mode()
    matrix = tensor_matricization(tensor, u_legs, v_legs,
                                  correctly_ordered=correctly_ordered)
    u, s, vh = np.linalg.svd(matrix, full_matrices=full_matrices)
    shape = tensor.shape
    u_shape = _determine_tensor_shape(shape, u, u_legs, output=True)
    vh_shape = _determine_tensor_shape(shape, vh, v_legs, output=False)
    u = np.reshape(u, u_shape)
    vh = np.reshape(vh, vh_shape)
    return u, s, vh

def _determine_tensor_shape(old_shape: Tuple[int,...],
                            matrix: np.ndarray,
                            legs: Tuple[int,...],
                            output: bool = True) -> Tuple[int,...]:
    """
    Determines the shape of a tensor after a decomposition.
    
    Determines the new shape a matrix is to be reshaped to after a decomposition
    of a tensor with some original shape, to again obtain a tensor.
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

@dataclass
class SVDParameters:
    """
    Holds all the parameters required for a truncated singular value
    decomposition.
    
    Attributes:
        max_bond_dim (int, optional): The maximum bond dimension allowed
            between nodes. Defaults to 100.
        rel_tol (float, optional): singular values s for which
            (s / largest singular value) < rel_tol are truncated. Defaults to
            0.01.
        total_tol (float, optional): singular values s for which s < total_tol
            are truncated. Defaults to 1e-15.
        renorm (bool, optional): If True, the truncated singular value vector
            is scaled to have the same norm as the original vector. Defaults to
            False.
        sum_trunc (bool, optional): If True, all singular values with index
            larger than K are truncated, where K is the smallest index such

            .. math::
                \sum_{i=K}^{r} (s_i / ||s|| )^2 < \text{total_tol}^2
            
            where r is the number of singular values. Defaults to False.
        
    """
    max_bond_dim: int = 100
    rel_tol: float = 0.01
    total_tol: float = 1e-15
    renorm: bool = False
    sum_trunc: bool = False
    sum_renorm: bool = True

    def __post_init__(self):
        """
        Check the validity of the parameters.
        """
        self.check_truncation_parameters()

    def check_truncation_parameters(self):
        """
        Checks if the truncation parameters are valid.

        The maximum bond dimension has to be a positive integer or infinity.
        The relative tolerance has to be positive or -infinity.
        The total tolerance has to be positive or -infinity.
        """
        max_bond_dim = self.max_bond_dim
        if (not isinstance(max_bond_dim,int)) and (max_bond_dim != float("inf")):
            raise TypeError(f"'max_bond_dim' has to be int not {type(max_bond_dim)}!")
        positivity_check(max_bond_dim, "max_bond_dim")
        rel_tol = self.rel_tol
        if (rel_tol < 0) and (rel_tol != float("-inf")):
            raise ValueError("'rel_tol' has to be positive or -inf.")
        total_tol = self.total_tol
        if (total_tol < 0) and (total_tol != float("-inf")):
            raise ValueError("'total_tol' has to be positive or -inf.")

def renormalise_singular_values(s: np.ndarray,
                                new_s: np.ndarray) -> np.ndarray:
    """
    Renormalises a truncated singular value vector.

    The vector is scaled to have the same norm as the original vector.

    Args:
        s (np.ndarray): The original vector of singular values.
        new_s (np.ndarray): The truncated vector of singular values.

    Returns:
        np.ndarray: The renormalised vector new_s.
    """
    norm_old = np.sum(s)
    norm_new = np.sum(new_s)
    new_s = new_s * norm_old / norm_new
    return new_s

def _sum_truncation_index(s: np.ndarray,
                          total_tol: float,
                          norming: bool = True) -> int:
    """
    Determines the index for truncation of the singular values.

    The index is determined by the condition

    .. math::
        \sum_{i=K}^{r} (s_i / ||s|| )^2 < \text{total_tol}^2

    where r is the number of singular values.

    Args:
        s (np.ndarray): Vector of singular values sorted in descending order.
        total_tol (float): Tolarance for the sum of the squared singular
            values.
        norming (bool, optional): If True, the sum of the squared singular
            values is normalised by the squared norm of the vector. Defaults to
            True.
    
    Returns:
        int: The index K for truncation.
    
    """
    normsq = np.linalg.norm(s)**2
    if normsq == 0:
        # In this case all singular values are zero, so all are truncated.
        return 0
    thresh = total_tol**2
    trunc_sum = 0
    for i, s_val in enumerate(reversed(s)):
        # Note that the singular values are sorted in descending order.
        trunc_sum += s_val**2
        if norming:
            comp_val = trunc_sum / normsq
        else:
            comp_val = trunc_sum
        if comp_val > thresh:
            return len(s) - i
    # In this case all singular values are truncated
    return 0

def sum_truncation(s: np.ndarray,
                   total_tol: float,
                   norming: bool = True
                   ) -> np.ndarray:
    """
    Truncates the singular values of a tensor given as a vector by according to

    .. math::
        \sum_{i=K}^{r} (s_i / ||s|| )^2 < \text{total_tol}^2

    where r is the number of singular values.

    Args:
        s (np.ndarray): Vector of singular values sorted in descending order.
        total_tol (float): Tolarance for the sum of the squared singular
            values.
        norming (bool, optional): If True, the truncated singular value vector
            is scaled to have the same norm as the original vector. Defaults to
            True.

    Returns:
        np.ndarray: The truncated vector of singular values.
    
    """
    trunc_index = _sum_truncation_index(s, total_tol, norming=norming)
    return s[:trunc_index]

def value_truncation(s: np.ndarray,
                     total_tol: float,
                     rel_tol: float) -> np.ndarray:
    """
    Truncates a vector of by removing all singular values that are smaller than

    .. math::
        \max(\text{rel_tol} \cdot \text{max}(s), \text{total_tol}).

    Args:
        s (np.ndarray): Vector of singular values sorted in descending order.
        total_tol (float): Absolute value tolerance for the singular values.
        rel_tol (float): Tolarance for the relative size of the singular values.

    Returns:
        np.ndarray: The truncated vector of singular values.

    """
    max_singular_value = s[0]
    min_singular_value_cutoff = max(rel_tol * max_singular_value,
                                    total_tol)
    s_temp = s[s > min_singular_value_cutoff]
    return s_temp

def truncate_singular_values(s: np.ndarray,
                             svd_params: SVDParameters) -> Tuple[np.ndarray,np.ndarray]:
    """
    Truncates the singular values of a tensor given as a vector.

    Args:
        s (np.ndarray): Vector of singular values sorted in descending order.
        svd_params (SVDParameters): Parameters for the truncation of the singular
            values.

    Returns:
        Tuple[np.ndarray,np.ndarray]: (new_s, s_trunc), where is is the
            shortened vector of singular values and s_trunc is a vector of the
            truncated singular values.
    """
    if len(s) == 0:
        raise ValueError("No singular values to truncate!")
    if svd_params.sum_trunc:
        s_temp = sum_truncation(s, svd_params.total_tol, svd_params.sum_renorm)
    else:
        s_temp = value_truncation(s, svd_params.total_tol, svd_params.rel_tol)
    max_bond_dim = svd_params.max_bond_dim
    if len(s_temp) > max_bond_dim:
        new_s = s_temp[:max_bond_dim]
        s_trunc = s[max_bond_dim:]
    elif len(s_temp) == 0:
        warn("All singular values were truncated. Returning only the largest singular value.")
        s_trunc = s[1:]
        new_s = np.array([s[0]])
    else:
        new_s = s_temp
        s_trunc = s[len(s_temp):]
    if svd_params.renorm:
        new_s = renormalise_singular_values(s, new_s)
    return new_s, s_trunc

def truncated_tensor_svd(tensor: np.ndarray,
                         u_legs: Tuple[int,...],
                         v_legs: Tuple[int,...],
                         svd_params: SVDParameters) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Performs a singular value decomposition of a tensor including truncation.

    This means some of the singular values are discarded and the resulting
    tensors are truncated to the new bond dimension.
    
    Args:
        tensor (np.ndarray): Tensor on which the svd is to be performed.
        u_legs (Tuple[int]): Legs of tensor that are to be associated to U
            after the SVD.
        v_legs (Tuple[int]): Legs of tensor that are to be associated to V
            after the SVD.
        svd_params (SVDParameters): Parameters for the truncation of the
            singular values.
    
    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray]: (U, S, V), where U and V are
         the truncated tensors and S are the singular values::

                  |2                             |1
                __|_      v_legs = (1, )       __|_        ____        ____
               |    |     q_legs = (0,2)      |    |      |  S |      |    |
            ___|    |___  -------------->  ___|  U |______|____|______| Vh |____1 
            0  |____|  1                   0  |____| 2   0       1  0 |____|

    """
    u, s, vh = tensor_svd(tensor, u_legs, v_legs)
    s, _ = truncate_singular_values(s, svd_params)
    new_bond_dim = len(s)
    u = u[..., :new_bond_dim]
    vh = vh[:new_bond_dim, ...]
    return u, np.asarray(s), vh

class ContractionMode(Enum):
    """
    Which tensor the singular values are contracted into.

    VCONTR: Contracts the singular values with the V tensor.
    UCONTR: Contracts the singular values with the U tensor.
    EQUAL: Contracts the squareroot of the singular values with each tensor.
    """
    UCONTR = "ucontr"
    VCONTR = "vcontr"
    EQUAL = "equal"

def contr_truncated_svd_splitting(tensor: np.ndarray,
                                  u_legs: Tuple[int,...],
                                  v_legs: Tuple[int,...],
                                  contr_mode: ContractionMode = ContractionMode.VCONTR,
                                  svd_params: SVDParameters = SVDParameters()) -> Tuple[np.ndarray,np.ndarray]:
    """
    Performs a truncated singular value decomposition, but the singular values
     are contracted with the V tensor.

    Args:
        tensor (np.ndarray): Tensor on which the svd is to be performed.
        u_legs (Tuple[int]): Legs of tensor that are to be associated to U
            after the SVD.
        v_legs (Tuple[int]): Legs of tensor that are to be associated to V
            after the SVD.
        contr_mode (ContractionMode): Determines how the singular values are
            contracted into the other tensors.
        svd_params (SVDParameters): Parameters for the truncation of the
    """
    u, s, vh = truncated_tensor_svd(tensor, u_legs, v_legs, svd_params)
    if contr_mode == ContractionMode.VCONTR:
        svh = np.tensordot(np.diag(s), vh, axes=(1, 0))
        return u, svh
    if contr_mode == ContractionMode.UCONTR:
        us = np.tensordot(u,np.diag(s), axes=(-1, 0))
        return us, vh
    if contr_mode == ContractionMode.EQUAL:
        s_root = np.diag(np.sqrt(s))
        svh = np.tensordot(s_root, vh, axes=(1, 0))
        us = np.tensordot(u,s_root,axes=(-1, 0))
        return us, svh
    raise ValueError("Invalid contraction mode!")

def idiots_splitting(tensor: np.ndarray,
                     a_legs: Tuple[int,...],
                     b_legs: Tuple[int,...],
                     a_tensor: Union[np.ndarray,None] = None,
                     b_tensor: Union[np.ndarray,None] = None) -> Tuple[np.ndarray,np.ndarray]:
    """
    An idiots splitting of a tensor by two given compatible tensors.

    Performs checks if the given tensors are compatible with the given legs.

    Args:
        tensor (np.ndarray): Tensor to be split.
        a_legs (Tuple[int]): Legs of tensor that are to be associated to A.
        b_legs (Tuple[int]): Legs of tensor that are to be associated to B.
        a_tensor (Union[np.ndarray,None], optional): Given tensor A. Leg
            connecting to B is the last leg. Defaults to None.
        b_tensor (Union[np.ndarray,None], optional): Given tensor B. Leg
            connecting to A is the first leg. Defaults to None.

    Returns:
        Tuple[np.ndarray,np.ndarray]: (A, B), the two split tensors.

    """
    if a_tensor is None or b_tensor is None:
        raise ValueError("Both tensors have to be given!")
    tensor_shape_a = tuple([tensor.shape[i] for i in a_legs])
    tensor_shape_b = tuple([tensor.shape[i] for i in b_legs])
    a_shape = a_tensor.shape[0:-1]
    b_shape = b_tensor.shape[1:]
    if tensor_shape_a != a_shape:
        raise ValueError("A tensor not compatible!")
    if tensor_shape_b != b_shape:
        raise ValueError("B tensor not compatible!")
    if a_tensor.ndim != len(a_legs) + 1:
        raise ValueError("A tensor has wrong number of legs!")
    if b_tensor.ndim != len(b_legs) + 1:
        raise ValueError("B tensor has wrong number of legs!")
    if a_tensor.shape[-1] != b_tensor.shape[0]:
        raise ValueError("A and B tensor not compatible!")
    return (a_tensor, b_tensor)
