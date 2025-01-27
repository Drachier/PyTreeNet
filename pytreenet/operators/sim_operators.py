"""
Functions to generate operators commonly used for example simulations.
"""
from typing import List, Dict, Union, Tuple

from numpy import ndarray

from .tensorproduct import TensorProduct
from .hamiltonian import Hamiltonian
from ..core.tree_structure import TreeStructure

def single_site_operators(operator: Union[str,ndarray],
                          node_identifiers: Union[TreeStructure,List[str]],
                          operator_names: Union[List[str],None]=None
                          ) -> Dict[str,TensorProduct]:
    """
    Define a constant operator on each node.

    Args:
        operator (Union[str,ndarray]): The operator to apply to each node.
        node_identifiers (Union[TreeStructure,List[str]]): The node identifiers
            for which the operators should be created. Can also be pulled from
            a TreeStructure object.
        operator_names (Union[List[str],None]): The names of the operators.
            If None, the operator names are set to the node identifiers.
        
    Returns:
        Dict[str,TensorProduct]: The operators.

    """
    if isinstance(node_identifiers, TreeStructure):
        node_identifiers = list(node_identifiers.nodes.keys())
    if operator_names is None:
        operator_names = node_identifiers
    else:
        assert len(operator_names) == len(node_identifiers)
    operators = {operator_names[i]: TensorProduct({node_identifiers[i]: operator})
                 for i in range(len(node_identifiers))}
    return operators

def create_nearest_neighbour_hamiltonian(structure: Union[TreeStructure,List[Tuple[str,str]]],
                                         local_operator1: Union[ndarray, str],
                                         local_operator2: Union[ndarray, str, None] = None,
                                         conversion_dict: Union[Dict[str,ndarray],None] = None
                                         ) -> Hamiltonian:
    """
    Creates a nearest neighbour Hamiltonian for a given tree structure and
     local operator.
    So for every nearest neighbour pair (i,j) the Hamiltonian will contain a
     term A_i (x) B_j, where A_i is the local operator at site i.
    
    Args:
        structure (Union[TreeStructure,List[Tuple[str,str]]]): The tree
            structure for which the Hamiltonian should be created or a list
            of tuples of nearest neighbours.
        local_operator1 (Union[ndarray, str]): The local operator to be used
            to generate the nearest neighbour interaction, i.e. A. Which is
            equal for all i.
        local_operator2 (Union[ndarray, str, None]): The local operator to be
            used to generate the nearest neighbour interaction, i.e. B. Which
            is equal for all j. If None will be the same as A.
        conversion_dict (Union[Dict[str,ndarray],None]): A conversion
            that can be used, if symbolic operators were used. Defaults to
            None.
    
    Returns:
        Hamiltonian: The Hamiltonian for the given structure.
    """
    if local_operator2 is None:
        local_operator2 = local_operator1
    if isinstance(structure, TreeStructure):
        structure = structure.nearest_neighbours()
    terms = []
    for identifier1, identifier2, in structure:
        terms.append(TensorProduct({identifier1: local_operator1,
                                    identifier2: local_operator2}))
    return Hamiltonian(terms, conversion_dictionary=conversion_dict)

def create_single_site_hamiltonian(structure: Union[TreeStructure,List[str]],
                                   local_operator: Union[str, ndarray],
                                   conversion_dict: Union[Dict[str,ndarray],None] = None
                                   ) -> Hamiltonian:
    """
    Creates a Hamiltonian for a given tree structure and local operators.
    The Hamiltonian will contain a term A for every site i
    
    Args:
        structure (Union[TreeStructure,List[str]]): The tree structure for
            which the Hamiltonian should be created.
        local_operator (Union[str, ndarray]): The local operators to be used
            to generate the single site interaction, i.e. A_i for every i.
        conversion_dict (Union[Dict[str,ndarray],None]): A conversion
            that can be used, if symbolic operators were used. Defaults to None.
    
    Returns:
        Hamiltonian: The Hamiltonian for the given structure.
    """
    terms = single_site_operators(local_operator, structure) # dictionary
    terms = list(terms.values()) # list of operators
    return Hamiltonian(terms, conversion_dictionary=conversion_dict)
