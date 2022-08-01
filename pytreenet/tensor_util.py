"""
Helpfull functions that work with the tensors of the tensor nodes.
"""

import numpy as np
from math import prod



def tensor_matricization(tensor, output_legs, input_legs):
    """
    Parameters
    ----------
    tensor : ndarray
        Tensor to be matricized.
    output_legs : tuple of int
        The tensor legs which are to be combined to be the matrix' output leg.
    input_legs : tuple of int
        The tensor legs which are to be combined to be the matrix' input leg.

    Returns
    -------
    matrix : ndarray
        The resulting matrix.
    """
    assert tensor.ndim == len(output_legs) + len(input_legs)

    correct_leg_order = output_legs + input_legs
    tensor_correctly_ordered = np.transpose(tensor, axes=correct_leg_order)
    shape = tensor_correctly_ordered.shape
    output_dimension = prod(shape[0:len(output_legs)])
    input_dimension = prod(shape[len(output_legs):])

    matrix = np.reshape(tensor_correctly_ordered, (output_dimension, input_dimension))
    return matrix

def determine_tensor_shape(old_shape, matrix, legs, output = True):
    """
    Determines the new shape a matrix is to be reshaped to after a decomposition
    of a tensor of old_shape.

    Parameters
    ----------
    old_shape : tuple of int
        Shape of the original tensor.
    legs : tuple of int
        Which legs of the original tensor are associated to matrix.
    ouput : boolean, optional
        If the legs of the original tensor are associated to the input or
        output of matrix. The default is True.

    Returns
    -------
    new_shape : tuple of int
        New shape to which matrix is to be reshaped.

    """
    leg_shape = [old_shape[i] for i in legs]
    if output:
        matrix_dimension = [matrix[1]]
        new_shape = leg_shape.extend(matrix_dimension)
    else:
        matrix_dimension = [matrix[0]]
        new_shape = matrix_dimension.extend(leg_shape)

    return new_shape

def tensor_qr_decomposition(tensor, q_legs, r_legs, mode='reduced'):
    """
    Parameters
    ----------
    tensor : ndarray
        Tensor on which the QR-decomp is to applied.
    q_legs : tuple of int
        Legs of tensor that should be associated to the Q tensor after
        QR-decomposition.
    r_legs : tuple of int
        Legs of tensor that should be associated to the R tensor after
        QR-decomposition.
    mode: {'reduced', 'full'}
        'reduced' returns the reduced QR-decomposition and 'full' the full QR-
        decomposition. Refer to the documentation of numpy.linalg.qr for
        further details. The default is 'full'.

    Returns
    -------
    q, r : ndarray
        The resulting tensors from the QR-decomposition. Refer to the
        documentation of numpy.linalg.qr for further details.

    """
    assert (mode == "reduced") or (mode == "full"), "Only 'reduced' and 'full' are acceptable values for mode."
    matrix = tensor_matricization(tensor, q_legs, r_legs)
    q, r = np.linalg.qr(matrix, mode=mode)
    shape = tensor.shape
    q_shape = determine_tensor_shape(shape, q, q_legs, output = True)
    r_shape = determine_tensor_shape(shape, r, r_legs, output = False)
    q = np.reshape(q, q_shape)
    r = np.reshape(r, r_shape)
    return q, r

def tensor_svd(tensor, u_legs, v_legs, mode='full'):
    """
    Parameters
    ----------
    tensor : ndarray
        Tensor on which the svd is to be performed.
    u_legs : tuple of int
        Legs of tensor that are to be associated to U after the SVD.
    v_legs : tuple of int
        Legs of tensor that are to be associated to V after the SVD.
    mode : {'fulll', 'reduced'}
        Determines if the full or reduced SVD is returned.
        The default is 'full'.

    Returns
    -------
    None.

    """
    assert (mode == "reduced") or (mode == "full"), "Only 'reduced' and 'full' are acceptable values for mode."

    # Cases to deal with input format of numpy function
    if mode == 'full':
        full_matrices = True
    elif mode == 'reduced':
        full_matrices = False

