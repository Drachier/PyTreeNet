"""
This module allows to generate random tensor products.

The randomly generated tensor products can be either symbolic or numeric.
They can also be generated from a reference tree or by providing a list of
possible operators.
"""
from typing import Union, List

import numpy as np
from numpy.random import default_rng, Generator

from ..core.ttn import TreeTensorNetwork
from ..operators.tensorproduct import TensorProduct
from .random_matrices import crandn
from ..util.ttn_exceptions import non_negativity_check, positivity_check

def random_tensor_product(ref_tree: TreeTensorNetwork,
                          num_operators: int = 1,
                          possible_operators: Union[List[str],List[np.ndarray],None] = None,
                          factor: float = 1.0,
                          seed: Union[int,Generator,None] = None) -> TensorProduct:
    """
    Generates a random tensor product that is compatible with the reference
     TreeTensorNetwork.

    Args:
        ref_tree (TreeTensorNetwork): A reference TreeTensorNetwork.
            It provides the identifiers and dimensions for the operators in the
            tensor product.
        num_operators (int, optional): The number of operators in the tensor
            product. These are the non-identity operators. Defaults to 1.
        possible_operators (Union[List[str],List[np.ndarray],None]): A list of
            possible operators that can be chosen as non-identity operators.
            These can be either strings or numpy arrays, defining a symbolic or
            numeric operator. If none are provided, the function will generate
            a numeric operator, infering the dimensions from the reference
            tree. Defaults to None.
        factor (float, optional): A factor that is multiplied to the operators.
            Ignored for symbolic operators. Defaults to 1.0.
        seed (Union[int,Generator,None], optional): A seed for the random number
            generator or a generator itself. Defaults to None.

    Returns:
        TensorProduct: The generated random tensor product.
    """
    non_negativity_check(num_operators, "number of operators")
    if num_operators > len(ref_tree.nodes):
        errstr = "There cannot be more non-trivial operators than nodes in the tree!"
        raise ValueError(errstr)
    if num_operators == 0:
        return TensorProduct()
    if possible_operators is None or len(possible_operators) == 0 or isinstance(possible_operators[0],np.ndarray):
        return random_numeric_tensor_product(ref_tree, num_operators,
                                             possible_operators, factor,
                                             seed)
    identifiers = list(ref_tree.nodes.keys())
    return random_symbolic_tensor_product(identifiers, possible_operators,
                                          num_operators, seed)

def random_symbolic_tensor_product(identifiers: List[str],
                                   possible_operators: List[str],
                                   num_operators: int = 1,
                                   seed: Union[int,Generator,None] = None) -> TensorProduct:
    """
    Generates a random symbolic tensor product.

    Args:
        identifiers (List[str]): A list of identifiers for the operators in the
            tensor product.
        possible_operators (List[str]): A list of possible operators that can be
            chosen as non-identity operators.
        num_operators (int, optional): The number of operators in the tensor
            product. These are the non-identity operators. Defaults to 1.
        seed (Union[int,Generator,None], optional): A seed for the random number
            generator or a generator itself. Defaults to None.
    
    Returns:
        TensorProduct: The generated random tensor product.
    """
    errstr = "There has to be at least one non-trivial operator to choose from!"
    positivity_check(len(possible_operators), errstr=errstr)
    if num_operators > len(identifiers):
        errstr = "There cannot be more non-trivial operators than identifiers!"
        raise ValueError(errstr)
    rng = default_rng(seed)
    random_identifiers = rng.choice(identifiers, size=num_operators,
                                    replace=False)
    random_operators = rng.choice(possible_operators, size=num_operators)
    return TensorProduct(dict(zip(random_identifiers, random_operators)))

def random_numeric_tensor_product(ref_tree: TreeTensorNetwork,
                                  num_operators: int = 1,
                                  possible_operators: Union[List[np.ndarray],None] = None,
                                  factor: float = 1.0,
                                  seed: Union[int,Generator,None] = None) -> TensorProduct:
    """
    Generates a random numeric tensor product from a reference tree.

    Args:
        ref_tree (TreeTensorNetwork): A reference TreeTensorNetwork.
            It provides the identifiers and potentially the operators for the
            tensor product.
        num_operators (int, optional): The number of operators in the tensor
            product. These are the non-identity operators. Defaults to 1.
        possible_operators (Union[List[np.ndarray],None], optional): A list of
            possible operators that can be chosen as non-identity operators.
            If none are provided, the function will generate a numeric operator,
            infering the dimensions from the reference tree. Defaults to None.
        factor (float, optional): A factor that is multiplied to the operators.
            Defaults to 1.0.
        seed (Union[int,Generator,None], optional): A seed for the random number
            generator or a generator itself. Defaults to None.
    
    Returns:
        TensorProduct: The generated random tensor product.
    """
    non_negativity_check(num_operators, "number of operators")
    if num_operators > len(ref_tree.nodes):
        errstr = "There cannot be more non-trivial operators than nodes in the tree!"
        raise ValueError(errstr)
    if possible_operators is None or len(possible_operators) == 0:
        return random_numeric_tensor_product_from_tree(ref_tree,
                                                       num_operators,
                                                       factor,
                                                       seed)
    identifiers = list(ref_tree.nodes.keys())
    return random_numeric_tensor_product_from_list(identifiers,
                                                   possible_operators,
                                                   num_operators,
                                                   factor,
                                                   seed)

def random_numeric_tensor_product_from_list(identifiers: List[str],
                                            possible_operators: List[np.ndarray],
                                            num_operators: int = 1,
                                            factor: float = 1.0,
                                            seed: Union[int,Generator,None] = None) -> TensorProduct:
    """
    Generates a random numeric tensor product.

    Args:
        identifiers (List[str]): A list of identifiers for the operators in the
            tensor product.
        possible_operators (List[np.ndarray]): A list of possible operators that
            can be chosen as non-identity operators.
        num_operators (int, optional): The number of operators in the tensor
            product. These are the non-identity operators. Defaults to 1.
        factor (float, optional): A factor that is multiplied to the operators.
            Defaults to 1.0.
        seed (Union[int,Generator,None], optional): A seed for the random number
            generator or a generator itself. Defaults to None.
    
    Returns:
        TensorProduct: The generated random tensor product.
    """
    errstr = "There has to be at least one non-trivial operator to choose from!"
    positivity_check(len(possible_operators), errstr=errstr)
    if num_operators > len(identifiers):
        errstr = "There cannot be more non-trivial operators than identifiers!"
        raise ValueError(errstr)
    rng = default_rng(seed)
    random_identifiers = rng.choice(identifiers, size=num_operators,
                                    replace=False)
    random_operators = rng.choice(possible_operators, size=num_operators)
    random_operators[0] = factor * random_operators[0]
    return TensorProduct(dict(zip(random_identifiers, random_operators)))

def random_numeric_tensor_product_from_tree(ref_tree: TreeTensorNetwork,
                                            num_operators: int = 1,
                                            factor: float = 1.0,
                                            seed: Union[int,Generator,None] = None) -> TensorProduct:
    """
    Generates a random numeric tensor product from a reference tree.

    Args:
        ref_tree (TreeTensorNetwork): A reference TreeTensorNetwork.
            It provides the identifiers and dimensions for the operators in the
            tensor product.
        num_operators (int, optional): The number of operators in the tensor
            product. These are the non-identity operators. Defaults to 1.
        factor (float, optional): A factor that is multiplied to the operators.
            Defaults to 1.0.
        seed (Union[int,Generator,None], optional): A seed for the random number
            generator or a generator itself. Defaults to None.
    """
    rng = default_rng(seed)
    chosen_nodes = rng.choice(list(ref_tree.nodes.values()), num_operators,
                              replace=False)
    identifiers = []
    operators = []
    for node in chosen_nodes:
        identifiers.append(node.identifier)
        operator = crandn((node.open_dimension(),node.open_dimension()))
        operators.append(operator)
    operators[0] = factor * operators[0]
    return TensorProduct(dict(zip(identifiers, operators)))
