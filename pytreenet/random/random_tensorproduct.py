from random import sample
from typing import Union, List

import numpy as np

from ..core.ttn import TreeTensorNetwork
from ..operators.tensorproduct import TensorProduct
from ..util.util import crandn

def random_tensor_product(reference_tree: TreeTensorNetwork,
                          num_operators: int = 1,
                          possible_operators: Union[List[str],List[np.ndarray]],
                          factor: float = 1.0) -> TensorProduct:
    """
    Generates a random tensor product that is compatible with the reference
     TreeTensorNetwork.

    Args:
        reference_tree (TreeTensorNetwork): A reference TreeTensorNetwork.
         It provides the identifiers and dimensions for the operators in the
         tensor product.
        num_factors (int): The number of factors to use. The nodes to which they
         are applied are drawn randomly from all nodes.
    """
    if num_operators < 0:
        errstr = "The number of factors must be non-negative!"
        errstr =+ f"{num_operators} < 1!"
        raise ValueError(errstr)
    if num_operators > len(reference_tree.nodes):
        errstr = "There cannot be more factors than nodes in the tree!"
        errstr =+ f"{num_operators} > {len(reference_tree.nodes)}!"
        raise ValueError(errstr)

    random_tp = TensorProduct()
    chosen_nodes = sample(list(reference_tree.nodes.values()), num_operators)
    for node in chosen_nodes:
        factor = crandn((node.open_dimension(),node.open_dimension()))
        random_tp[node.identifier] = factor
    return random_tp
