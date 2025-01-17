"""
This module supplies all functions to generate random Hamiltonians.

The Hamiltonians can be both symbolic and numeric.
"""
from typing import List, Union, Tuple, Dict

from numpy.random import default_rng, Generator
from numpy import ndarray, eye
from fractions import Fraction
from enum import Enum

from ..core.ttn import TreeStructure
from ..operators.tensorproduct import TensorProduct
from ..operators.hamiltonian import Hamiltonian
from .random_matrices import random_hermitian_matrix
from .random_ttns import random_big_ttns_two_root_children


class RandomType(Enum):
    """
    Enum class to specify the type of random Hamiltonian to be generated.
    """
    UNIQUE = "Unique"
    RANDOM = "Random"

def random_hamiltonian(num_of_terms: int,
                       possible_operators: Union[List[str],List[ndarray]],
                       tree: TreeStructure,
                       strength: Tuple[float,float] = (-1,1),
                       num_sites: Tuple[int,int] = (2,2),
                       conversion_dict: Union[None,Dict[str,ndarray]] = None,
                       seed: Union[None,int,Generator] = None) -> Hamiltonian:
    """
    Generates a random Hamiltonian.

    The function generates a Hamiltonian with a given number of terms. The
    operators acting in each term are randomly chosen.

    Args:
        num_of_terms (int): The number of terms in the Hamiltonian.
        possible_operators (Union[List[str],List[ndarray]]): A list of all
            possible single site operators. The operators can be symbolic or
            numerical.
        tree (TreeStructure): The tree structure that the Hamiltonian should be
            compatible with.
        strength (Tuple[float,float], optional): The range of strengths that
            the interaction terms can have. Defaults to (-1,1).
        num_sites (Tuple[int,int], optional): The range of the number of sites
            that can partake in an interaction term. This means the number of
            sites that have one of the possible operators applied to them.
            Defaults to (2,2).
        conversion_dict (Union[None,Dict[str,ndarray]], optional): A dictionary
            that maps the symbolic operators to numerical matrices. Defaults to
            None.
    
    Returns:
        Hamiltonian: A random Hamiltonian.
    """
    identifiers = list(tree.nodes.keys())
    rand_terms = random_terms(num_of_terms,
                              possible_operators,
                              identifiers,
                              min_strength=strength[0],
                              max_strength=strength[1],
                              min_num_sites=num_sites[0],
                              max_num_sites=num_sites[1],
                              seed=seed)
    return Hamiltonian(rand_terms,
                       conversion_dictionary=conversion_dict)

def random_terms(num_of_terms: int,
                 possible_operators: Union[List[str],List[ndarray]],
                 sites: List[str],
                 min_strength: float = -1,
                 max_strength: float = 1,
                 min_num_sites: int = 2,
                 max_num_sites: int = 2,
                 seed: Union[None,int,Generator] = None,
                 w_coeff: Union[None,bool] = False) -> List[TensorProduct]:
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
        seed (Union[None,int,Generator], optional): A seed for the random
            number generator or a generator itself. Defaults to None.
    
    Returns:
        List[TensorProduct]: A list containing all the random terms.
    """
    if isinstance(possible_operators[0], str):
        if w_coeff:
            return random_symbolic_terms_with_coeffs(
                                         num_of_terms, possible_operators, 
                                         sites, min_strength, max_strength, 
                                         min_num_sites, max_num_sites, seed)
        else:
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
        seed (Union[None,int,Generator], optional): A seed for the random
            number generator or a generator itself. Defaults to None.
    
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
                        seed: Union[None,int,Generator] = None ) -> TensorProduct:
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
        seed (Union[None,int,Generator], optional): A seed for the random
            number generator or a generator itself. Defaults to None.

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
        seed (Union[None,int,Generator], optional): A seed for the random
            number generator or a generator itself. Defaults to None.

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
        sites (list[str]): Identifiers of the nodes to which they may be
            applied.
        num_sites (int, optional): Number of non-trivial sites in a term.
            Defaults to 2.
        seed (Union[int, None], optional): A seed for the random number
            generator. Defaults to None.

    Returns:
        TensorProduct: A random term in the form of a tensor product
    """
    rng = default_rng(seed=seed)
    num_sites = rng.integers(low=min_num_sites, high=max_num_sites)
    rand_sites = rng.choice(sites, size=num_sites, replace=False)
    rand_operators = rng.choice(possible_operators, size=num_sites)
    return TensorProduct(dict(zip(rand_sites, rand_operators)))

def random_symbolic_terms_with_coeffs(num_of_terms: int,
                          possible_operators: List[str],
                          sites: List[str],
                          min_coeff: float = -10,
                          max_coeff: float = 10,
                          min_num_sites: int = 2,
                          max_num_sites: int = 2,
                          random_type: RandomType = RandomType.UNIQUE,
                          seed=None,
                          possible_gammas: Union[List[str],None] = None
                          ) -> List[TensorProduct]:
    """
    Creates random symbolic interaction terms with random coefficients.

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
        seed (Union[None,int,Generator], optional): A seed for the random
            number generator or a generator itself. Defaults to None.

    Returns:
        List[TensorProduct]: A list containing all the random terms.
    """
    rterms = []
    coeffs = []
    rng = default_rng(seed=seed)
    for i in range(num_of_terms):
        term = random_symbolic_term(possible_operators, sites,
                                    min_num_sites=min_num_sites,
                                    max_num_sites=max_num_sites,
                                    seed=seed)
        
        if random_type == RandomType.UNIQUE:
            assert len(possible_gammas.keys()) > num_of_terms, "Not enough possible gammas for unique terms"
            coeff_gamma = possible_gammas.keys()[i]
        elif random_type == RandomType.RANDOM:
            coeff_gamma = rng.choice(list(possible_gammas.keys()))

        coeff_lambda = Fraction(rng.choice([i for i in range(min_coeff, max_coeff) if i != 0]), rng.integers(1, 4))
        while abs(possible_gammas[coeff_gamma] * float(coeff_lambda)) <= 1.5 or abs(possible_gammas[coeff_gamma] * float(coeff_lambda)) >= 10:
            coeff_lambda = Fraction(rng.choice([i for i in range(min_coeff, max_coeff) if i != 0]), rng.integers(1, 4))
            
        while term in rterms: # To avoid multiples
            term = random_symbolic_term(possible_operators, sites,
                                        min_num_sites=min_num_sites,
                                        max_num_sites=max_num_sites,
                                        seed=seed)
            
        rterms.append((coeff_lambda,coeff_gamma,term))
        
    return rterms

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
