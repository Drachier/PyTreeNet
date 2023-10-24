from __future__ import annotations
from typing import Dict, Union
from numpy.random import default_rng
from numpy import asarray, ndarray
from enum import Enum, auto

from .ttn_exceptions import NotCompatibleException
from .operators.operator import NumericOperator
from .operators.tensorproduct import TensorProduct
from .util import compare_lists_by_value

class PadMode(Enum):
    risky = auto()
    safe = auto()


class Hamiltonian(object):
    """
    Represents the Hamiltonian on a TTN.
    The entries of the main list should represent the terms of the Hamiltonian
    and be a dictionary. The key is an identifier which could be matched to a
    TensorNode in the TTN and the value is an operator that is to be applied to
    that node/site.
    """

    def __init__(self, terms: list[TensorProduct] = None,
                 conversion_dictionary: Dict[str, ndarray] = None):
        """
        Initialises a Hamiltonian from a number of terms represented by a TensorProduct each:
            H = sum( terms )

        Args:
            terms (list[TensorProduct], optional): A list of TensorProduct making up the
             Hamiltonian. Defaults to None.
            conversion_dictionary (dict, optional): A conversion dictionary might be supplied.
             It is used, if the tensor products are symbolic. Defaults to None.
        """
        if terms is None:
            self.terms = []
        else:
            self.terms = terms

        if conversion_dictionary is None:
            self.conversion_dictionary = {}
        else:
            self.conversion_dictionary = conversion_dictionary

    def __repr__(self):
        return str(self.terms)

    def __eq__(self, other_hamiltonian):
        """
        Two Hamiltonians are equal, if all of their terms are equal.
        """
        return compare_lists_by_value(self.terms, other_hamiltonian.terms)

    def __add__(self, other: Union[TensorProduct, Hamiltonian]) -> Hamiltonian:
        if isinstance(other, TensorProduct):
            self.add_term(other)
        elif isinstance(other, Hamiltonian):
            self.add_hamiltonian(other)
        else:
            errstr = f"Addition between Hamiltonian and {type(other)} not supported!"
            raise TypeError(errstr)
        return self

    def add_term(self, term: TensorProduct):
        """
        Adds a term to the Hamiltonian.

        Args:
            term (TensorProduct): The term to be added in the form of a TensorProduct
        """
        self.terms.append(term)

    def add_hamiltonian(self, other: Hamiltonian):
        """
        Adds one Hamiltonian to this Hamiltonian. The other Hamiltonian will not be modified.

        Args:
            other (Hamiltonian): Hamiltonian to be added.
        """
        self.terms.extend(other.terms)
        self.conversion_dictionary.update(other.conversion_dictionary)

    def add_multiple_terms(self, terms: list[TensorProduct]):
        """
        Add multiple terms to this Hamiltonian

        Args:
            terms (list[TensorProduct]): Terms to be added.
        """
        self.terms.extend(terms)

    def is_compatible_with(self, ttn: TreeTensorNetwork) -> bool:
        """
        Returns, if the Hamiltonian is compatible with the provided TTN. Compatibility means
         that all node identifiers that appear any term of this Hamiltonian are identifiers
         of nodes in the TTN

        Args:
            ttn (TreeTensorNetwork): The TTN to check against.

        Returns:
            bool: Whether the two are compatible or not.
        """
        for term in self.terms:
            for site_id in term:
                if not site_id in ttn.nodes:
                    return False
        return True
    
    def perform_compatibility_checks(self, mode: PadMode,
                                     reference_ttn: TreeTensorNetwork):
        """
        Performs the check of the mode and the check of compatibility, if desired.

        Args:
            mode (PadMode, optional): 'safe' performs a compatability check with the reference
             ttn. Risky will not run this check, which might be time consuming for large
             TTN. Defaults to PadMode.safe.

        Raises:
            NotCompatibleException: If the Hamiltonian and TTN are not compatible
            ValueError: If a wrong mode is used.
        """
        if mode == PadMode.safe:
            if not self.is_compatible_with(reference_ttn):
                raise NotCompatibleException(
                    "Hamiltonian and reference_ttn are incompatible")
        elif mode != PadMode.risky:
            raise ValueError(
                f"{mode} is not a valid option for 'mode'. (Only 'safe' and 'risky are)!")

    def pad_with_identities(self, reference_ttn: TreeTensorNetwork,
                          mode: PadMode = PadMode.safe, 
                          symbolic: bool = True) -> Hamiltonian:
        """
        Returns a Hamiltonian, where all terms are padded with an identity according to
         the refereence reference_ttn.

        Args:
            reference_ttn (TreeTensorNetwork): Provides the structure on which padding should
             occur. Furthermore the dimension of the open legs of each provide the new
             identities' dimensions.
            mode (PadMode, optional): 'safe' performs a compatability check with the reference
             ttn. Risky will not run this check, which might be time consuming for large
             TTN. Defaults to PadMode.safe.
            symbolic (bool, optional): Whether the terms should be padded with a symbolic
             identity or an actual array. Defaults to True.

        Raises:
            NotCompatibleException: If the Hamiltonian and TTN are not compatible
            ValueError: If a wrong mode is used.
        """
        self.perform_compatibility_checks(mode=mode, reference_ttn=reference_ttn)
        new_terms = []
        for term in self.terms:
            new_term = term.pad_with_identities(reference_ttn, symbolic=symbolic)
            new_terms.append(new_term)
        return Hamiltonian(new_terms, conversion_dictionary=self.conversion_dictionary)

    def to_matrix(self, ref_ttn: TreeTensorNetwork, use_padding: bool = True,
                  mode: PadMode = PadMode.safe) -> NumericOperator:
        """
        Creates a Numeric operator that is equivalent to the Hamiltonian.
         The resulting operator can get very large very fast, so this should only be used
         for debugging. The result is a matrix valued operator.

        Args:
            ref_ttn (TreeTensorNetwork): TTN giving the tree structure which the
             Hamiltonian should respect.
            use_padding (bool, optional): Enable, if the Hamiltonian requires padding with
             respect to the reference TTN. Defaults to True.
            mode (PadMode, optional): 'safe' performs a compatability check with the reference
             ttn. Risky will not run this check, which might be time consuming for large
             TTN. Defaults to PadMode.safe.

        Returns:
            NumericOperator: Operator corresponding to the Hamiltonian.
        """
        self.perform_compatibility_checks(mode=mode, reference_ttn=ref_ttn)
        if use_padding:
            self.pad_with_identities(ref_ttn)
        full_tensor = asarray([0], dtype=complex)
        identifiers = list(ref_ttn.nodes.keys())
        for i, term in enumerate(self.terms):
            term_operator = term.into_operator(conversion_dict=self.conversion_dictionary,
                                               order=identifiers)
            if i == 0:
                full_tensor = term_operator.operator
            else:
                full_tensor += term_operator.operator
        return NumericOperator(full_tensor.T, identifiers)

    def to_tensor(self, ref_ttn: TreeTensorNetwork, use_padding: bool = True,
                  mode: PadMode = PadMode.safe) -> NumericOperator:
        """
        Creates a NumericOperator that is equivalent to the Hamiltonian.
         The resulting operator can get very large very fast, so this should only be used
         for debugging. The result is a tensor with multiple legs.

        Args:
            ref_ttn (TreeTensorNetwork): TTN giving the tree structure which the
             Hamiltonian should respect.
            use_padding (bool, optional): Enable, if the Hamiltonian requires padding with
             respect to the reference TTN. Defaults to True.
            mode (PadMode, optional): 'safe' performs a compatability check with the reference
             ttn. Risky will not run this check, which might be time consuming for large
             TTN. Defaults to PadMode.safe.

        Returns:
            NumericOperator: Operator corresponding to the Hamiltonian.
        """
        matrix_operator = self.to_matrix(ref_ttn,use_padding=use_padding,mode=mode)
        shape = [node.open_dimension() for node in ref_ttn.nodes.values()]
        shape *= 2
        tensor_operator = matrix_operator.operator.reshape(shape)
        return NumericOperator(tensor_operator, matrix_operator.node_identifiers)

    def contains_duplicates(self) -> bool:
        """
        Checks, if there are duplicates of terms. Can be especially important after padding.

        Returns:
            bool: True if there are duplicates, False otherwise
        """
        dup = [term for term in self.terms if self.terms.count(term) > 1]
        return len(dup) > 0

def random_terms(
        num_of_terms: int, possible_operators: list, sites: list[str],
        min_strength: float = -1, max_strength: float = 1, min_num_sites: int = 2,
        max_num_sites: int = 2):
    """
    Creates random interaction terms.

    Parameters
    ----------
    num_of_terms : int
        The number of random terms to be generated.
    possible_operators : list of arrays
        A list of all possible single site operators. We assume all sites have
        the same physical dimension.
    sites : list of str
        A list containing the possible identifiers of site nodes.
    min_strength : float, optional
        Minimum strength an interaction term can have. The strength is
        multiplied to the first operator of the term. The default is -1.
    max_strength : float, optional
        Minimum strength an interaction term can have. The strength is
        multiplied to the first operator of the term. The default is 1.
    min_num_sites : int, optional
        The minimum numberof sites that can partake in a single interaction
        term. The default is 2.
    max_num_sites : int, optional
        The minimum numberof sites that can partake in a single interaction
        term. The default is 2.

    Returns
    -------
    rterms : list of dictionaries
        A list containing all the random terms.
    """

    rterms = []

    rng = default_rng()
    number_of_sites = rng.integers(low=min_num_sites, high=max_num_sites + 1,
                                   size=num_of_terms)
    strength = rng.uniform(low=min_strength, high=max_strength,
                           size=num_of_terms)

    for index, nsites in enumerate(number_of_sites):
        term = {}
        operator_indices = rng.integers(len(possible_operators), size=nsites)
        sites_list = []
        first = True

        for operator_index in operator_indices:

            operator = possible_operators[operator_index]

            if first:
                # The first operator has the interaction strength
                operator = strength[index] * operator
                first = False

            site = sites[rng.integers(len(sites))]
            # Every site should appear maximally once (Good luck)
            while site in sites_list:
                site = sites[rng.integers(len(sites))]

            term[site] = operator

        rterms.append(term)

    return rterms


def random_symbolic_terms(num_of_terms: int, possible_operators: list[ndarray], sites: list[str],
                          min_num_sites: int = 2,  max_num_sites: int = 2, seed=None):
    """
    Creates random interaction terms.

    Parameters
    ----------
    num_of_terms : int
        The number of random terms to be generated.
    possible_operators : list of arrays
        A list of all possible single site operators. We assume all sites have
        the same physical dimension.
    sites : list of str
        A list containing the possible identifiers of site nodes.
    min_num_sites : int, optional
        The minimum numberof sites that can partake in a single interaction
        term. The default is 2.
    max_num_sites : int, optional
        The minimum numberof sites that can partake in a single interaction
        term. The default is 2.

    Returns
    -------
    rterms : list of dictionaries
        A list containing all the random terms.
    """
    rterms = []
    rng = default_rng(seed=seed)
    for _ in range(num_of_terms):
        number_of_sites = rng.integers(low=min_num_sites,
                                        high=max_num_sites + 1,
                                        size=num_of_terms)
        term = random_symbolic_term(possible_operators, sites,
                                    num_sites=number_of_sites,
                                    seed=rng)
        while term in rterms:
            term = random_symbolic_term(possible_operators, sites,
                                        num_sites=number_of_sites,
                                        seed=rng)
        rterms.append(term)
    return rterms


def random_symbolic_term(possible_operators: list[str], sites: list[str],
                         num_sites: int = 2, seed: Union[int, None]=None) -> TensorProduct:
    """
    Creates a random interaction term.

    Args:
        possible_operators (list[ndarray]): Symbolic operators to choose from.
        sites (list[str]): Identifiers of the nodes to which they may be applied.
        num_sites (int, optional): Number of non-trivial sites in a term. Defaults to 2.
        seed (Union[int, None], optional): A seed for the random number generator. Defaults to None.

    Returns:
        TensorProduct: A random term in the form of a tensor product
    """
    rng = default_rng(seed=seed)
    rand_sites = rng.choice(sites, size=num_sites, replace=False)
    rand_operators = rng.choice(possible_operators, size=num_sites)
    return TensorProduct(dict(zip(rand_sites, rand_operators)))
