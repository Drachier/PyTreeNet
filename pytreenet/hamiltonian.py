from __future__ import annotations
from typing import Union
from numpy.random import default_rng
from numpy import prod, eye, tensordot

from .ttn_exceptions import NotCompatibleException

from enum import Enum, auto


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

    def __init__(self, terms: list[TensorProduct] = None, conversion_dictionary: dict = None):
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

    def __add__(self, other: Union[TensorProduct, Hamiltonian]):
        if isinstance(other, TensorProduct):
            self.add_term(other)
        elif isinstance(other, Hamiltonian):
            self.add_hamiltonian(other)
        else:
            errstr = f"Addition between Hamiltonian and {type(other)} not supported!"
            raise TypeError(errstr)

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
        if mode == PadMode.safe:
            if not self.is_compatible_with(reference_ttn):
                raise NotCompatibleException(
                    "Hamiltonian and reference_ttn are incompatible")
        elif mode != PadMode.risky:
            raise ValueError(
                f"{mode} is not a valid option for 'mode'. (Only 'safe' and 'risky are)!")
        for term in self.terms:
            term.pad_with_identities(reference_ttn, symbolic=symbolic)

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

    def to_tensor(self, ref_ttn: TreeTensorNetwork, use_padding: bool = False):
        """
        Creates a tensor ndarray representing this Hamiltonian assuming it is
        defined on the structure of ttn.

        Parameters
        ----------
        ref_ttn : TreeTensorNetwork
            TTN giving the tree structure which the Hamiltonian should respect.

        Returns
        -------
        full_tensor: ndarray
            A tensor representing the Hamiltonian. Every node in the ref_ttn
            corresponds to two legs in the tensor.

        """
        if use_padding:
            self.pad_with_identity(ref_ttn)

        first = True
        for term in self.terms:

            term_tensor = self.conversion_dictionary[term[ref_ttn.root_id]]
            term_tensor = self._to_tensor_rec(ref_ttn,
                                              ref_ttn.root_id,
                                              term,
                                              term_tensor)

            if first:
                full_tensor = term_tensor
                first = False
            else:
                full_tensor += term_tensor

        # Separating input and output legs
        permutation = list(range(0, full_tensor.ndim, 2))
        permutation.extend(list(range(1, full_tensor.ndim, 2)))
        full_tensor = full_tensor.transpose(permutation)
        return full_tensor

    def _to_tensor_rec(self, ttn: TreeTensorNetwork, node_id: str, term: dict, tensor: TensorNode):
        for child_id in ttn.nodes[node_id].children:
            child_tensor = self.conversion_dictionary[term[child_id]]
            tensor = tensordot(tensor, child_tensor, axes=0)
            tensor = self._to_tensor_rec(ttn, child_id, term, tensor)

        return tensor

    def to_matrix(self, ttn: TreeTensorNetwork):
        """
        Creates a matrix ndarray representing this Hamiltonian assuming it is
        defined on the structure of ttn.

        Parameters
        ----------
        ttn : TreeTensorNetwork
            TTN giving the tree structure which the Hamiltonian should respect.

        Returns
        -------
        matrix: ndarray
            A matrix representing the Hamiltonian.

        """
        matrix = self.to_tensor(ttn)
        half_dim = matrix.ndim / 2
        matrix_size = prod(matrix.shape[0:half_dim])

        return matrix.reshape((matrix_size, matrix_size))

    def contains_duplicates(self):
        """
        If the there are equal terms contained. Especially important to recheck
        after padding.

        Returns
        -------
        result: bool

        """
        dup = [term for term in self.terms if self.terms.count(term) > 1]
        return len(dup) > 0

    def __add__(self, other_hamiltonian: Hamiltonian):

        total_terms = []
        total_terms.extend(self.terms)
        total_terms.extend(other_hamiltonian.terms)

        new_hamiltonian = Hamiltonian(terms=total_terms)
        return new_hamiltonian

    def __repr__(self):
        return str(self.terms)

    def __eq__(self, other_hamiltonian):
        """
        Two Hamiltonians are equal, if all of their terms are equal.
        """
        return self.terms == other_hamiltonian.terms


def random_terms(
        num_of_terms: int, possible_operators: list, sites: list[str],
        min_strength: float = -1, max_strength: float = 1, min_num_sites: int = 2, max_num_sites: int = 2):
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
    number_of_sites = rng.integers(low=min_num_sites, high=max_num_sites + 1,
                                   size=num_of_terms)

    for num_sites in number_of_sites:
        term = random_symbolic_term(possible_operators, sites,
                                    num_sites=num_sites, seed=rng)

        while term in rterms:
            term = random_symbolic_term(possible_operators, sites,
                                        num_sites=num_sites, seed=rng)

        rterms.append(term)

    return rterms


def random_symbolic_term(possible_operators: list[ndarray], sites: list[str], num_sites: int = 2, seed=None):
    """
    Creates a random interaction term.

    Parameters
    ----------
    possible_operators : list of arrays
        A list of all possible single site operators. We assume all sites have
        the same physical dimension.
    sites : list of str
        A list containing the possible identifiers of site nodes.
    num_sites : int, optional
        The number of sites that are non-trivial in this term. The default is 2.

    Returns
    -------
    rterm : dict
        A dictionary containing the sites as keys and the symbolic operators
        as value.
    """
    rng = default_rng(seed=seed)
    rand_sites = rng.choice(sites, size=num_sites, replace=False)
    rand_operators = rng.choice(possible_operators, size=num_sites)

    return dict(zip(rand_sites, rand_operators))
