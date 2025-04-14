"""
Functions to generate operators commonly used for example simulations.
"""
from typing import List, Dict, Union, Tuple
from fractions import Fraction

from numpy import ndarray

from .tensorproduct import TensorProduct
from .hamiltonian import Hamiltonian
from ..core.tree_structure import TreeStructure

def single_site_operators(operator: Union[str,ndarray],
                          node_identifiers: Union[TreeStructure,List[str]],
                          factor: Union[List[tuple[Fraction,str]],tuple[Fraction,str],None]=None,
                          operator_names: Union[List[str],None]=None
                          ) -> Dict[str,Tuple[Fraction,str,TensorProduct]]:
    """
    Define a constant operator on each node.

    Args:
        operator (Union[str,ndarray]): The operator to apply to each node.
        node_identifiers (Union[TreeStructure,List[str]]): The node identifiers
            for which the operators should be created. Can also be pulled from
            a TreeStructure object.
        factor (Union[List[tuple[Fraction,str]],tuple[Fraction,str],None]):
            The factors to multiply the operator with. If given a list
            the order of factors must be the same as the order of identifiers.
            If None, the factor isset to 1.
        operator_names (Union[List[str],None]): The names of the operators.
            If None, the operator names are set to the node identifiers.
        
    Returns:
        Dict[str,TensorProduct]: The operators.

    """
    if isinstance(node_identifiers, TreeStructure):
        node_identifiers = list(node_identifiers.nodes.keys())
    if factor is None:
        factor = [(Fraction(1), "1")] * len(node_identifiers)
    elif factor[0] == 0:
        # Factually zero operators
        return {}
    elif isinstance(factor, tuple):
        factor = [factor] * len(node_identifiers)
    assert len(factor) == len(node_identifiers)
    if operator_names is None:
        operator_names = node_identifiers
    else:
        assert len(operator_names) == len(node_identifiers)
    operators = {operator_names[i]: (factor[i][0],factor[i][1],TensorProduct({node_identifiers[i]: operator}))
                 for i in range(len(node_identifiers))}
    return operators

def create_nearest_neighbour_hamiltonian(structure: Union[TreeStructure,List[Tuple[str,str]]],
                                         local_operator1: Union[ndarray, str],
                                         factor: Union[tuple[Fraction,str],None] = None,
                                         local_operator2: Union[ndarray, str, None] = None,
                                         conversion_dict: Union[Dict[str,ndarray],None] = None,
                                         coeffs_mapping: Union[Dict[str,complex],None] = None
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
        coeffs_mapping (Union[Dict[str,complex],None]): A mapping of the
            coefficients to be used for the operators. Defaults to None.
    
    Returns:
        Hamiltonian: The Hamiltonian for the given structure.
    """
    if factor is None:
        factor = (Fraction(1), "1")
    elif factor[0] == 0:
        # Factually a zero Hamiltonian
        return Hamiltonian([],
                           conversion_dictionary=conversion_dict,
                           coeffs_mapping=coeffs_mapping)
    if local_operator2 is None:
        local_operator2 = local_operator1
    if isinstance(structure, TreeStructure):
        structure = structure.nearest_neighbours()
    terms = []
    for identifier1, identifier2, in structure:
        term_op = TensorProduct({identifier1: local_operator1,
                                    identifier2: local_operator2})
        terms.append((factor[0], factor[1], term_op))
    return Hamiltonian(terms,
                       conversion_dictionary=conversion_dict,
                       coeffs_mapping=coeffs_mapping)

def create_single_site_hamiltonian(structure: Union[TreeStructure,List[str]],
                                   local_operator: Union[str, ndarray],
                                   factor: Union[tuple[Fraction,str],None] = None,
                                   conversion_dict: Union[Dict[str,ndarray],None] = None,
                                   coeffs_mapping: Union[Dict[str,complex],None] = None
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
        coeffs_mapping (Union[Dict[str,complex],None]): A mapping of the
            coefficients to be used for the operators. Defaults to None.
    
    Returns:
        Hamiltonian: The Hamiltonian for the given structure.
    """
    terms = single_site_operators(local_operator,
                                  structure,
                                  factor=factor) # dictionary
    terms = list(terms.values()) # list of operators
    return Hamiltonian(terms,
                       conversion_dictionary=conversion_dict,
                       coeffs_mapping=coeffs_mapping)
