"""
Helpfull functions that work with the tensors of the tensor nodes.
"""

import numpy as np
from math import prod
import scipy.sparse as sps

def transpose_tensor_by_leg_list(tensor, first_legs, last_legs):
    """
    Transposes a tensor according to two lists of legs. All legs in first_legs
    will become the first legs of the new tensor and the last_legs will all
    become the last legs of the tensor. 
    
    Parameters
    ----------
    tensor : ndarray
        Tensor to be transposed.
     first_legs : list of int
        Leg indices that are to become the first legs of the new tensor.
    last_legs : list of int
        Leg indices that are to become the last legs of the new tensor..

    Returns
    -------
    transposed_tensor: ndarray
        New tensor that is the transposed input tensor.

    """
    assert tensor.ndim == len(first_legs) + len(last_legs)
    
    correct_leg_order = first_legs + last_legs
    transposed_tensor = np.transpose(tensor, axes=correct_leg_order)
    return transposed_tensor

def tensor_matricization(tensor, output_legs, input_legs,
                         correctly_ordered=False):
    """
    Parameters
    ----------
    tensor : ndarray
        Tensor to be matricized.
    output_legs : tuple of int
        The tensor legs which are to be combined to be the matrix' output leg.
    input_legs : tuple of int
        The tensor legs which are to be combined to be the matrix' input leg.
    correctly_ordered: bool, optional
        If true it is assumed, the tensor does not need to be transposed, i.e.
        this should be activated if the tensor already has the correct order
        of legs. Else 

    Returns
    -------
    matrix : ndarray
        The resulting matrix.
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

    matrix = np.reshape(tensor_correctly_ordered, (output_dimension, input_dimension))
    return matrix

def _determine_tensor_shape(old_shape, matrix, legs, output = True):
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
        matrix_dimension = [matrix.shape[1]]
        leg_shape.extend(matrix_dimension)
        new_shape = leg_shape
    else:
        matrix_dimension = [matrix.shape[0]]
        matrix_dimension.extend(leg_shape)
        new_shape = matrix_dimension

    return tuple(new_shape)

def tensor_qr_decomposition(tensor, q_legs, r_legs, mode='complete', keep_shape=False):
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
    mode: {'reduced', 'complete'}
        'reduced' returns the reduced QR-decomposition and 'complete' the complete QR-
        decomposition. Refer to the documentation of numpy.linalg.qr for
        further details. The default is 'complete'.

    Returns
    -------
    q, r : ndarray
        The resulting tensors from the QR-decomposition. Refer to the
        documentation of numpy.linalg.qr for further details.

    """
    assert (mode == "reduced") or (mode == "complete"), "Only 'reduced' and 'complete' are acceptable values for mode."
    matrix = tensor_matricization(tensor, q_legs, r_legs)

    if not keep_shape:
        q, r = np.linalg.qr(matrix, mode=mode)
    else:
        u, s_vec, vh = np.linalg.svd(matrix, full_matrices=True)
        s_mat = np.zeros((matrix.shape[1], matrix.shape[1]))
        s_mat[:s_vec.shape[0], :s_vec.shape[0]] = np.diag(s_vec)
        if u.shape[1] < matrix.shape[1]:
            u_new = np.zeros(matrix.shape, dtype=complex)
            u_new[:, :u.shape[1]] = u
            u = u_new
        elif u.shape[1] > matrix.shape[1]:
            u_new = u[:, :matrix.shape[1]]
            u = u_new
        q = u
        r = np.matmul(s_mat, vh)

    shape = tensor.shape
    q_shape = _determine_tensor_shape(shape, q, q_legs, output = True)
    r_shape = _determine_tensor_shape(shape, r, r_legs, output = False)
    q = np.reshape(q, q_shape)
    r = np.reshape(r, r_shape)
    return q, r

def tensor_svd(tensor, u_legs, v_legs, mode='reduced'):
    """
    Parameters
    ----------
    tensor : ndarray
        Tensor on which the svd is to be performed.
    u_legs : tuple of int
        Legs of tensor that are to be associated to U after the SVD.
    v_legs : tuple of int
        Legs of tensor that are to be associated to V after the SVD.
    mode : {'complete', 'reduced'}
        Determines if the full or reduced matrices u and vh for the SVD are
        returned. The default is 'reduced'.

    Returns
    -------
    u : ndarray
        The unitary array contracted to the left of the diagonalised singular
        values.
    s : ndarray
        A vector consisting of the tensor's singular values.
    vh : ndarray
        The unitary array contracted to the right of the diagonalised singular
        values.

    """
    if (mode != "reduced") and (mode != "complete"):
        raise ValueError(f"'mode' may only be 'complete' or 'reduced' not {mode}!")

    # Cases to deal with input format of numpy function
    if mode == 'complete':
        full_matrices = True
    elif mode == 'reduced':
        full_matrices = False

    matrix = tensor_matricization(tensor, u_legs, v_legs)
    u, s, vh = np.linalg.svd(matrix, full_matrices=full_matrices)
    shape = tensor.shape
    u_shape = _determine_tensor_shape(shape, u, u_legs, output = True)
    vh_shape = _determine_tensor_shape(shape, vh, v_legs, output = False)
    u = np.reshape(u, u_shape)
    vh = np.reshape(vh, vh_shape)

    return u, s, vh

def check_truncation_parameters(max_bond_dim, rel_tol, total_tol):
    if (type(max_bond_dim) != int) and (max_bond_dim != float("inf")):
        raise TypeError(f"'max_bond_dim' has to be int not {type(max_bond_dim)}!")
    elif max_bond_dim < 0:
        raise  ValueError("'max_bond_dim' has to be positive.")
    elif (rel_tol < 0) and (rel_tol != float("-inf")):
        raise ValueError("'rel_tol' has to be positive or -inf.")
    elif (total_tol < 0) and (total_tol != float("-inf")):
        raise ValueError("'total_tol' has to be positive or -inf.")        

def truncated_tensor_svd(tensor, u_legs, v_legs,
                         max_bond_dim=100, rel_tol=0.01, total_tol=1e-15):
    """
    Performs an svd including truncation, i.e. discarding some singular values.
    
    Parameters
    ----------
    tensor : ndarray
        Tensor on which the svd is to be performed.
    u_legs : tuple of int
        Legs of tensor that are to be associated to U after the SVD.
    v_legs : tuple of int
        Legs of tensor that are to be associated to V after the SVD.
    max_bond_dim: int
        The maximum bond dimension allowed between nodes. Default is 100.
    rel_tol: float
        singular values s for which ( s / largest singular value) < rel_tol
        are truncated. Default is 0.01.
    total_tol: float
        singular values s for which s < total_tol are truncated.
        Defaults to 1e-15.

    Returns
    -------
    u : ndarray
        The unitary array contracted to the left of the diagonalised singular
        values.
    s : ndarray
        A vector consisting of the tensor's singular values.
    vh : ndarray
        The unitary array contracted to the right of the diagonalised singular
        values.
    """
    check_truncation_parameters(max_bond_dim, rel_tol, total_tol)
    
    matrix = tensor_matricization(tensor, u_legs, v_legs)
    u, s, vh = np.linalg.svd(matrix)
    
    # Here the truncation happens
    max_singular_value = s[0]
    min_singular_value_cutoff = max(rel_tol * max_singular_value, total_tol)
    s = [singular_value for singular_value in s
         if singular_value > min_singular_value_cutoff]
    
    if len(s) > max_bond_dim:
        s = s[:max_bond_dim]
    elif len(s) == 0:
        s = [max_singular_value]
    
    new_bond_dim = len(s)
    u = u[:,:new_bond_dim]
    vh = vh[:new_bond_dim,:]
    
    shape = tensor.shape
    u_shape = _determine_tensor_shape(shape, u, u_legs, output = True)
    vh_shape = _determine_tensor_shape(shape, vh, v_legs, output = False)
    u = np.reshape(u, u_shape)
    vh = np.reshape(vh, vh_shape)

    return u, s, vh
    
def set_leg_dimension(node1, index1, node2, index2, value=10, mode="set"):
    bond_dim = node1.tensor.shape[index1]
    if mode == "add":
        value += bond_dim

    tensor = np.tensordot(node1.tensor, node2.tensor, axes=(index1,index2))
    all_legs = [i for i in range(tensor.ndim)]
    node1_legs = [i for i in range(node1.tensor.ndim-1)]
    node2_legs = [i for i in all_legs if i not in node1_legs]
    u, s, vh = tensor_svd(tensor, node1_legs, node2_legs)

    if mode == "analyze":
        if s.shape[0] < bond_dim:
            s_ = np.zeros((bond_dim,))
            s_[:s.shape[0]] += s
            return s_
        return s

    q = np.tensordot(u, np.diag(s), axes=(-1, 0))
    r = vh

    if value > q.shape[-1]:
        q_new = np.zeros(q.shape[:-1]+(value,), dtype=q.dtype)
        q_new[..., :q.shape[-1]] += q
        q = q_new
    q = q[..., :value]
    axes = tuple(range(0, index1)) + (q.ndim-1,) + tuple(range(index1, q.ndim-1))
    q = q.transpose(axes)
    node1.tensor = q

    if value > r.shape[0]:
        r_new = np.zeros((value,)+r.shape[1:], dtype=r.dtype)
        r_new[:r.shape[0], ...] += r
        r = r_new
    r = r[:value, ...]
    axes = tuple(range(1, index2+1)) + (0,) + tuple(range(index2+1, r.ndim))
    r = r.transpose(axes)
    node2.tensor = r

def compute_transfer_tensor(tensor, open_indices):
    """

    Parameters
    ----------
    tensor : ndarray
    
    open_indices: tuple of int
        The open indices of tensor.
    
    Returns
    -------
    transfer_tensor : ndarry
        The transfer tensor of tensor, i.e., the tensor that contracts all
        open indices of tensor with the open indices of the tensor's complex
        conjugate.

    """
    conj_tensor = np.conjugate(tensor)
    transfer_tensor = np.tensordot(tensor, conj_tensor, 
                                   axes=(open_indices, open_indices))
    return transfer_tensor

def tensordot(a, b, axes, shape=None, mode="debug"):
    """
    Note_ This is compeltely ot used, but the code is still here.
    Perform the tensordot operation between tensors a and b along the legs 
    given by axes. Return output with specified shape.
    In essence: Perform
        tensor = np.tensordot(a, b, axes=axes)
        tensor = tensor_np.reshape(shape)

    Parameters
    ----------
    a: ndarray
        Tensor a.
    b: ndarray
        Tensor b.
    axes: tuple
        Legs along which the contraction is perfomed.
    mode: str
        One of
            "numpy": Perform np.tensordot(a, b, axes)
            "fast": Reduced type-checking and without redundant reshapes.
            "debug": Peform both versions and check if they are equal. 

    Returns
    -------
    tensor: ndarray
        Result of the tensor contraction. 
    """

    if mode == "numpy" or mode == "debug":
        tensor_np = np.tensordot(a, b, axes=axes)
        if shape is not None:
            tensor_np = tensor_np.reshape(shape)
    if mode == "fast" or mode == "debug":
        axes_a, axes_b = axes
        try:
            na = len(axes_a)
            axes_a = list(axes_a)
        except TypeError:
            axes_a = [axes_a]
            na = 1
        try:
            axes_b = list(axes_b)
        except TypeError:
            axes_b = [axes_b]

        as_ = a.shape
        nda = a.ndim
        bs = b.shape
        ndb = b.ndim

        for k in range(na):
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb

        # Move the axes to sum over to the end of "a"
        # and to the front of "b"
        notin = [k for k in range(nda) if k not in axes_a]
        newaxes_a = notin + axes_a
        N2 = 1
        for axis in axes_a:
            N2 *= as_[axis]
        newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
        olda = [as_[axis] for axis in notin]

        notin = [k for k in range(ndb) if k not in axes_b]
        newaxes_b = axes_b + notin
        N2 = 1
        for axis in axes_b:
            N2 *= bs[axis]
        newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
        oldb = [bs[axis] for axis in notin]

        at = a.transpose(newaxes_a).reshape(newshape_a)
        bt = b.transpose(newaxes_b).reshape(newshape_b)
        tensor_fast = np.dot(at, bt)

        if shape is None:
            shape = olda + oldb
        tensor_fast = tensor_fast.reshape(shape)

    if mode == "numpy":
        return tensor_np
    elif mode == "fast":
        return tensor_fast
    elif mode == "debug":
        assert np.allclose(tensor_np, tensor_fast), "tensordot error"
        return tensor_np
    else:
        raise ValueError


def tensor_multidot(tensor, other_tensors, main_legs, other_legs):
    """
    For a given tensor, perform multiple tensor contractions at once.
    
    Parameters
    ----------
    tensor : ndarray

    other_tensors : list (of ndarray)
        The tensors that should be contracted with tensor.

    main_legs : list (of int)
        The legs of tensor which are connected to the tensors in other_tensors.

    other_legs : list (of int)
        The legs of the tensors in other_tensors which are connected to tensor.
    
    Returns
    -------
    tensor: ndarray

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

def sparse_tensordot(a, b, axes=(0,0)):
    """
    Due to the sparcity during TDVP, this version of
    tensordot achieves a speed-up compared to Numpys
    tensordot.
    """
    ax_a, ax_b = axes

    notin_a = [a.shape[i] for i in range(a.ndim) if i!=ax_a]
    if a.ndim > 1:
        a = np.moveaxis(a, ax_a, 0).reshape(a.shape[ax_a], -1)
    else:
        a = np.reshape(a, (a.shape[ax_a], 1))
    a = sps.csr_matrix(a)

    notin_b = [b.shape[i] for i in range(b.ndim) if i!=ax_b]
    if b.ndim > 1:
        b = np.moveaxis(b, ax_b, 0).reshape(b.shape[ax_b], -1)
    else:
        b = np.reshape(b, (b.shape[ax_b], 1))
    b = sps.csr_matrix(b)

    results = a.T.dot(b).toarray().reshape(notin_a + notin_b)
    return results