"""
Functions to generate operators commonly used for example simulations.
"""
from typing import List, Dict, Union, Tuple
from fractions import Fraction
from warnings import warn

from numpy import ndarray

from .tensorproduct import TensorProduct
from .hamiltonian import Hamiltonian
from ..core.ttn import TreeTensorNetwork

def single_site_operators(operator: Union[str,ndarray],
                          node_identifiers: Union[TreeTensorNetwork,List[str]],
                          factor: Union[List[tuple[Fraction,str]],tuple[Fraction,str],None]=None,
                          operator_names: Union[List[str],None]=None,
                          with_factor: bool = True
                          ) -> Dict[str,Tuple[Fraction,str,TensorProduct]]:
    """
    Define a constant operator on each node.

    Args:
        operator (Union[str,ndarray]): The operator to apply to each node.
        node_identifiers (Union[TreeTensorNetwork,List[str]]): The node identifiers
            for which the operators should be created. Can also be pulled from
            a TreeTensorNetwork object.
        factor (Union[List[tuple[Fraction,str]],tuple[Fraction,str],None]):
            The factors to multiply the operator with. If given a list
            the order of factors must be the same as the order of identifiers.
            If None, the factor isset to 1.
        operator_names (Union[List[str],None]): The names of the operators.
            If None, the operator names are set to the node identifiers.
        with_factor (bool): If True, the factor is included in the operator.
            If False, the factor is not included in the operator.
        
    Returns:
        Dict[str,Tuple[Fraction,str,TensorProduct]]: The operators.

    """
    if isinstance(node_identifiers, TreeTensorNetwork):
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
    if with_factor:
        operators = {operator_names[i]: (factor[i][0],factor[i][1],TensorProduct({node_identifiers[i]: operator}))
                     for i in range(len(node_identifiers))}
    else:
        warnstr = "Do not use `with_factor=False`, instead use `create_single_site_observables`!"
        warn(warnstr, DeprecationWarning)
        operators = {operator_names[i]: TensorProduct({node_identifiers[i]: operator})
                     for i in range(len(node_identifiers))}
    return operators

def create_single_site_observables(operator: Union[str,ndarray],
                                   node_identifiers: Union[TreeTensorNetwork,List[str]],
                                   operator_names: Union[List[str],None]=None,
                          ) -> Dict[str,TensorProduct]:
    """
    Define a constant observable for each node.

    Observables do not have a factor associated to them.

    Args:
        operator (Union[str,ndarray]): The operator to apply to each node.
        node_identifiers (Union[TreeTensorNetwork,List[str]]): The node identifiers
            for which the operators should be created. Can also be pulled from
            a TreeTensorNetwork object.
        operator_names (Union[List[str],None]): The names of the operators.
            If None, the operator names are set to the node identifiers.
        
    Returns:
        Dict[str,TensorProduct]: The operators.

    """
    ops = single_site_operators(operator,
                                node_identifiers,
                                operator_names=operator_names)
    return {name: op[2] for name, op in ops.items()}

def create_nearest_neighbour_hamiltonian(structure: Union[TreeTensorNetwork,List[Tuple[str,str]]],
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
        structure (Union[TreeTensorNetwork,List[Tuple[str,str]]]): The tree
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
    if isinstance(structure, TreeTensorNetwork):
        if local_operator2 is not None:
            warn("Tree structure does not give a known order for nearest neighbours!")
        structure = structure.nearest_neighbours(consider_open=True)
    if local_operator2 is None:
        local_operator2 = local_operator1
    terms = []
    for identifier1, identifier2, in structure:
        term_op = TensorProduct({identifier1: local_operator1,
                                    identifier2: local_operator2})
        terms.append((factor[0], factor[1], term_op))
    return Hamiltonian(terms,
                       conversion_dictionary=conversion_dict,
                       coeffs_mapping=coeffs_mapping)

def create_single_site_hamiltonian(structure: Union[TreeTensorNetwork,List[str]],
                                   local_operator: Union[str, ndarray],
                                   factor: Union[tuple[Fraction,str],None] = None,
                                   conversion_dict: Union[Dict[str,ndarray],None] = None,
                                   coeffs_mapping: Union[Dict[str,complex],None] = None
                                   ) -> Hamiltonian:
    """
    Creates a Hamiltonian for a given tree structure and local operators.
    The Hamiltonian will contain a term A for every site i
    
    Args:
        structure (Union[TreeTensorNetwork,List[str]]): The tree structure for
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

def create_constant_hamiltonian(structure: list[list[str]],
                                oprator: str,
                                factor: tuple[Fraction, str] = (Fraction(1), "1"),
                                conversion_dict: dict[str, ndarray] | None = None,
                                coeffs_mapping: dict[str, complex] | None = None
                                ) -> Hamiltonian:
    """
    Create a constant Hamiltonian for a given structure.

    Args:
        structure (list[list[str]]): A list of node identifier combinations.
            For every combination a term of local operators will be created.
        oprator (ndarray | str): The local operator to be applied.
        factor (tuple[Fraction, str]): An optional factor to be associated to
            every term. Defaults to `(Fraction(1), "1")`.
        conversion_dict (dict[str, ndarray]): Will be added to the Hamiltonian
            if supplied. Otherwise it will be empty.
        coeffs_mapping: (dict[str, ndarray]): Will be added to the Hamiltonian
            if supplied. Otherwise it will be empty.
    
    Returns:
        Hamiltonian: The desired constant Hamiltonian.
    
    """
    if conversion_dict is None:
        conversion_dict = {}
    if coeffs_mapping is None:
        coeffs_mapping = {}
    ham = Hamiltonian(conversion_dictionary=conversion_dict,
                      coeffs_mapping=coeffs_mapping)
    for combi in structure:
        term = TensorProduct()
        for node_id in combi:
            term.add_operator(node_id, oprator)
        ham.add_term((factor[0], factor[1], term))
    return ham

def create_multi_site_hamiltonian(structure: list[list[str]],
                                  operators: list[str] | str,
                                  factor: tuple[Fraction, str] = (Fraction(1),"1"),
                                  conversion_dict: dict[str, ndarray] | None = None,
                                  coeffs_mapping: dict[str, complex] | None = None
                                  ) -> Hamiltonian:
    """
    Create a multi site hamiltonian.

    Args:
        structure (list[list[str]]): A list of node identifier combinations.
            For every combination a term of local operators will be created.
        operators (list[str] | str): The local operators to be applied. If a
            list, then the operators are applied to the nodes in the order of
            this list and the identifiers order. If only one operator, this
            operator will be applied to all nodes.
        factor (tuple[Fraction, str]): An optional factor to be associated to
            every term. Defaults to `(Fraction(1), "1")`.
        conversion_dict (dict[str, ndarray]): Will be added to the Hamiltonian
            if supplied. Otherwise it will be empty.
        coeffs_mapping: (dict[str, ndarray]): Will be added to the Hamiltonian
            if supplied. Otherwise it will be empty.

    Returns:
        Hamiltonian: The desired multi-site Hamiltonian.

    Raises:
        ValueError: If the different structure terms are not of the same length
            or the operator combination is of different length to the
            structure.
    
    """
    op_length = len(structure[0])
    if isinstance(operators, str):
        operators = op_length * [operators]
    else:
        if len(operators) != op_length:
            errstr = "Operator and supplied structre are incompatible!"
            raise ValueError(errstr)
    if conversion_dict is None:
        conversion_dict = {}
    if coeffs_mapping is None:
        coeffs_mapping = {}
    ham = Hamiltonian(conversion_dictionary=conversion_dict,
                      coeffs_mapping=coeffs_mapping)
    for combi in structure:
        if len(combi) != op_length:
            raise ValueError
        term = TensorProduct()
        for i, node_id in enumerate(combi):
            term.add_operator(node_id, operators[i])
        ham.add_term((factor[0], factor[1], term))
    return ham
