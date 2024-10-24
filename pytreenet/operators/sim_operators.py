"""
Functions to generate operators commonly used for example simulations.
"""
from typing import List, Dict, Union

from numpy import ndarray

from .tensorproduct import TensorProduct
from ..core.tree_structure import TreeStructure

def single_site_operators(operator: Union[str,ndarray],
                          node_identifiers: List[str],
                          operator_names: Union[List[str],None]=None
                          ) -> Dict[str,TensorProduct]:
    """
    Define a constant operator on each node.

    Args:
        operator (Union[str,ndarray]): The operator to apply to each node.
        node_identifiers (List[str]): The node identifiers.
        operator_names (Union[List[str],None]): The names of the operators.
            If None, the operator names are set to the node identifiers.
        
    Returns:
        Dict[str,TensorProduct]: The operators.

    """
    if operator_names is None:
        operator_names = node_identifiers
    else:
        assert len(operator_names) == len(node_identifiers)
    operators = {operator_names[i]: TensorProduct({node_identifiers[i]: operator})
                 for i in range(len(node_identifiers))}
    return operators

def single_site_operator_all_sites(operator: Union[str,ndarray],
                                   ttns: TreeStructure
                                   ) -> Dict[str,TensorProduct]:
    """
    Associate a single site operator to every site of a TTNS.
    """
    node_identifiers = [identifier for identifier in ttns.nodes.keys()]
    return single_site_operators(operator, node_identifiers)
