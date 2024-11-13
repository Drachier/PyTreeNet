from typing import List, TYPE_CHECKING
import numpy as np
from copy import deepcopy
from ..random.random_matrices import crandn
from ..contractions.state_operator_contraction import expectation_value
from ..contractions.state_state_contraction import contract_two_ttns
from ..ttno import TTNO
from ..ttns.ttns import TreeTensorNetworkState

def _init_orthogonal_states(tensor: np.ndarray, num_states: int, list_tensor: List[np.ndarray]=None)-> List[np.ndarray]:
    """
    Initialize orthonormal states for a multi-state node.
    
    Args:
        tensor: Input tensor that defines the desired shape
        num_states: Number of orthonormal states to generate
        
    Returns:
        list_tensor: List of orthonormal tensors with the same shape as input tensor
    """
    if list_tensor is None:
        list_tensor = [tensor]
        for _ in range(num_states-1):
            list_tensor.append(crandn(tensor.shape))
        # orthogonalize and normalize the states using the Gram-Schmidt process
    for i in range(num_states):
        # orthogonalize with respect to previous vectors
        for j in range(i):
            list_tensor[i] -= np.tensordot(list_tensor[i], list_tensor[j].conj(), 
                                         axes=(range(tensor.ndim), range(tensor.ndim))) * list_tensor[j]
        # normalize
        norm = np.sqrt(np.tensordot(list_tensor[i], list_tensor[i].conj(), 
                                  axes=(range(tensor.ndim), range(tensor.ndim))))
        if norm > 1e-10:  # avoid division by zero
            list_tensor[i] /= norm
    
    return list_tensor

def test_init_orthogonal_states():
    """
    Test the initialization of orthonormal states for a multi-state node.
    """
    tensor = crandn((4,6,8))
    list_tensor = _init_orthogonal_states(tensor, 5)
    # Check orthonormality: should print 1.0 for i==j, and close to 0 for i!=j
    for i, t1 in enumerate(list_tensor):
        for j, t2 in enumerate(list_tensor):
            inner_product = np.tensordot(t1, t2.conj(), axes=(range(tensor.ndim), range(tensor.ndim)))
            expected = 1.0 if i == j else 0.0
            if abs(inner_product - expected) > 1e-10:
                assert False, f"States {i},{j}: {inner_product:.6f} (should be {expected})"
    print("Test init_orthogonal_states passed!")
    

def _scalar_product_multittn(state_tensors: List[np.ndarray], weight: np.ndarray)-> complex:
    """
    Compute the scalar product of a multi-state tree tensor network state.
    """
    scalar_product = 0
    for i in range(len(state_tensors)):
        scalar_product += weight[i]*complex(np.tensordot(state_tensors[i], state_tensors[i].conj(), 
                                                        axes=(range(state_tensors[i].ndim), range(state_tensors[i].ndim))))
    return scalar_product

def test_scalar_product_multittn():
    """
    Test the scalar product of a multi-state tree tensor network state.
    """
    tensor = crandn((4,6,8))
    list_tensor = _init_orthogonal_states(tensor, 5)
    weight = np.array([1,1,1,1,1])
    scalar_product = _scalar_product_multittn(list_tensor, weight)
    assert scalar_product.real - weight.sum() < 1e-10, "Scalar product should be the sum of the weights"
    print("Test scalar_product_multittn passed!")
    