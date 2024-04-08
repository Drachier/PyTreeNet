from typing import List, Union, Dict

from numpy.random import default_rng, Generator
from numpy import ndarray, eye

from ..operators.tensorproduct import TensorProduct
from ..operators.hamiltonian import Hamiltonian
from .random_matrices import random_hermitian_matrix
from .random_ttns import random_big_ttns_two_root_children

def random_terms(num_of_terms: int,
                 possible_operators: Union[List[str],List[ndarray]],
                 sites: List[str],
                 min_strength: float = -1,
                 max_strength: float = 1,
                 min_num_sites: int = 2,
                 max_num_sites: int = 2,
                 seed: Union[None,int,Generator] = None) -> List[TensorProduct]:
    """
    Generates random interaction terms.

    The function generates a given number of interaction terms from a list of
    possible operators. The operators can be symbolic or numerical and a random
    strength can be assigned to each term.

    Args:
        num_of_terms (int): The number of random terms to be generated.
        possible_operators (Union[List[str],List[ndarray]]): A list of all
            possible single site operators.
        sites (List[str]): A list containing the possible identifiers of
            sites/nodes.
        min_strength (float, optional): Minimum strength an interaction term
            can have. The strength is multiplied to the first operator of the
            term. Defaults to -1 and is ignored for symbolic operators.
        max_strength (float, optional): Maximum strength an interaction term
            can have. The strength is multiplied to the first operator of the
            term. Defaults to 1 and is ignored for symbolic operators.
        min_num_sites (int, optional): The minimum number of sites that can
            partake in an interaction term, i.e. have one of the possible
            operators applied to them. Defaults to 2.
        max_num_sites (int, optional): The maximum number of sites that can
            partake in an interaction term, i.e. have one of the possible
            operators applied to them. Defaults to 2.
        seed (Union[None,int,Generator], optional): A seed for the random number
            generator or a generator itself. Defaults to None.
    
    Returns:
        List[TensorProduct]: A list containing all the random terms.
    """
    if isinstance(possible_operators[0], str):
        return random_symbolic_terms(num_of_terms, possible_operators, sites,
                                     min_num_sites, max_num_sites, seed)
    if isinstance(possible_operators[0], ndarray):
        return random_numeric_terms(num_of_terms, possible_operators, sites,
                                    min_strength, max_strength,
                                    min_num_sites, max_num_sites, seed)

def random_numeric_terms(num_of_terms: int,
                         possible_operators: List[ndarray],
                         sites: List[str],
                         min_strength: float = -1,
                         max_strength: float = 1,
                         min_num_sites: int = 2,
                         max_num_sites: int = 2,
                         seed: Union[None,int,Generator] = None) -> List[TensorProduct]:
    """
    Generates random numeric interaction terms.

    The function generates a given number of interaction terms from a list of
    possible matrices. A random strength can be assigned to each term.

    Args:
        num_of_terms (int): The number of random terms to be generated.
        possible_operators (Union[List[str],List[ndarray]]): A list of all
            possible single site operators.
        sites (List[str]): A list containing the possible identifiers of
            sites/nodes.
        min_strength (float, optional): Minimum strength an interaction term
            can have. The strength is multiplied to the first operator of the
            term. Defaults to -1 and is ignored for symbolic operators.
        max_strength (float, optional): Maximum strength an interaction term
            can have. The strength is multiplied to the first operator of the
            term. Defaults to 1 and is ignored for symbolic operators.
        min_num_sites (int, optional): The minimum number of sites that can
            partake in an interaction term, i.e. have one of the possible
            operators applied to them. Defaults to 2.
        max_num_sites (int, optional): The maximum number of sites that can
            partake in an interaction term, i.e. have one of the possible
            operators applied to them. Defaults to 2.
        seed (Union[None,int,Generator], optional): A seed for the random number
            generator or a generator itself. Defaults to None.
    
    Returns:
        List[TensorProduct]: A list containing all the random terms.
    """
    rterms = []
    for _ in range(num_of_terms):
        rterm = random_numeric_term(possible_operators,
                                    sites,
                                    min_strength,max_strength,
                                    min_num_sites,max_num_sites,
                                    seed=seed)
        rterms.append(rterm)
    return rterms

def random_numeric_term(possible_operators: List[ndarray],
                        sites: List[str],
                        min_strength: float = -1,
                        max_strength: float = 1,
                        min_num_sites: int = 2,
                        max_num_sites: int = 2,
                        seed: Union[None,int,Generator] = None) -> TensorProduct:
    """
    Generate a single random numeric interaction term.

    Args:
        possible_operators (Union[List[str],List[ndarray]]): A list of all
            possible single site operators.
        sites (List[str]): A list containing the possible identifiers of
            sites/nodes.
        min_strength (float, optional): Minimum strength an interaction term
            can have. The strength is multiplied to the first operator of the
            term. Defaults to -1 and is ignored for symbolic operators.
        max_strength (float, optional): Maximum strength an interaction term
            can have. The strength is multiplied to the first operator of the
            term. Defaults to 1 and is ignored for symbolic operators.
        min_num_sites (int, optional): The minimum number of sites that can
            partake in an interaction term, i.e. have one of the possible
            operators applied to them. Defaults to 2.
        max_num_sites (int, optional): The maximum number of sites that can
            partake in an interaction term, i.e. have one of the possible
            operators applied to them. Defaults to 2.
        seed (Union[None,int,Generator], optional): A seed for the random number
            generator or a generator itself. Defaults to None.

    Returns:
        TensorProduct: A random term in the form of a tensor product with
            matrices as single site operators.
    """
    rng = default_rng(seed=seed)
    num_sites = rng.integers(low=min_num_sites,  high=max_num_sites + 1)
    strength = rng.uniform(low=min_strength, high=max_strength)
    rand_sites = rng.choice(sites, size=num_sites, replace=False)
    rand_operators = rng.choice(possible_operators, size=num_sites)
    rand_operators[0] = strength * rand_operators[0]
    return TensorProduct(dict(zip(rand_sites, rand_operators)))

def random_symbolic_terms(num_of_terms: int,
                          possible_operators: List[str],
                          sites: List[str],
                          min_num_sites: int = 2,
                          max_num_sites: int = 2,
                          seed=None) -> List[TensorProduct]:
    """
    Creates random symbolic interaction terms.

    Args:
        num_of_terms (int): The number of random terms to be generated.
        possible_operators (List[str]): A list of all possible single site
            operators.
        sites (List[str]): A list containing the possible identifiers of
            sites/nodes.
        min_num_sites (int, optional): The minimum number of sites that can
            partake in an interaction term, i.e. have one of the possible
            operators applied to them. Defaults to 2.
        max_num_sites (int, optional): The maximum number of sites that can
            partake in an interaction term, i.e. have one of the possible
            operators applied to them. Defaults to 2.
        seed (Union[None,int,Generator], optional): A seed for the random number
            generator or a generator itself. Defaults to None.

    Returns:
        List[TensorProduct]: A list containing all the random terms.
    """
    rterms = []
    for _ in range(num_of_terms):
        term = random_symbolic_term(possible_operators, sites,
                                    min_num_sites=min_num_sites,
                                    max_num_sites=max_num_sites,
                                    seed=seed)
        while term in rterms: # To avoid multiples
            term = random_symbolic_term(possible_operators, sites,
                                        min_num_sites=min_num_sites,
                                        max_num_sites=max_num_sites,
                                        seed=seed)
        rterms.append(term)
    return rterms

def random_symbolic_term(possible_operators: List[str],
                         sites: List[str],
                         min_num_sites: int = 2,
                         max_num_sites: int = 2,
                         seed: Union[None,int,Generator] = None) -> TensorProduct:
    """
    Generates a random symbolic interaction term.

    Args:
        possible_operators (list[ndarray]): Symbolic operators to choose from.
        sites (list[str]): Identifiers of the nodes to which they may be applied.
        num_sites (int, optional): Number of non-trivial sites in a term. Defaults to 2.
        seed (Union[int, None], optional): A seed for the random number generator. Defaults to None.

    Returns:
        TensorProduct: A random term in the form of a tensor product
    """
    rng = default_rng(seed=seed)
    num_sites = rng.integers(low=min_num_sites, high=max_num_sites)
    rand_sites = rng.choice(sites, size=num_sites, replace=False)
    rand_operators = rng.choice(possible_operators, size=num_sites)
    return TensorProduct(dict(zip(rand_sites, rand_operators)))

def random_hamiltonian_compatible() -> Hamiltonian:
    """
    Generates a Hamiltonian that is compatible with the TTNS produced by
     `ptn.ttns.random_big_ttns_two_root_children`. It is already padded with
     identities.

    Returns:
        Hamiltonian: A Hamiltonian to use for testing.
    """
    conversion_dict = {chr(i): random_hermitian_matrix()
                       for i in range(65,70)} # A, B, C, D, E
    conversion_dict["I2"] = eye(2)
    terms = [TensorProduct({"site1": "A", "site2": "B", "site0": "C"}),
             TensorProduct({"site4": "A", "site3": "D", "site5": "C"}),
             TensorProduct({"site4": "A", "site3": "B", "site1": "A"}),
             TensorProduct({"site0": "C", "site6": "E", "site7": "C"}),
             TensorProduct({"site2": "A", "site1": "A", "site6": "D"}),
             TensorProduct({"site1": "A", "site3": "B", "site5": "C"})]
    ham = Hamiltonian(terms, conversion_dictionary=conversion_dict)
    ref_tree = random_big_ttns_two_root_children()
    return ham.pad_with_identities(ref_tree)
