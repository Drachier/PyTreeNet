from numpy.random import default_rng
from numpy import prod, eye, tensordot, reshape, transpose

from .ttn_exceptions import NotCompatibleException

class Hamiltonian(object):
    """
    Represents the Hamiltonian on a TTN.
    The entries of the main list should represent the terms of the Hamiltonian
    and be a dictionary. The key is an identifier which could be matched to a
    TensorNode in the TTN and the value is an operator that is to be applied to
    that node/site.
    """

    def __init__(self, terms=None):
        """
        Parameters
        ----------
        terms : list of dictionaries, optional
            A list of dictionaries containing the terms of the Hamiltonian. The
            keys are identifiers of the site to which the value, an operator,
            is to be applied.
            The default is None.

        """
        if terms == None:
            self.terms = []
        else:
            self.terms = terms

    def add_term(self, term):
        if not (type(term) == dict):
            try:
                term = dict(term)
            except TypeError:
                raise TypeError(f"{term} cannot be converted to a dictionary.")

        self.terms.append(term)

    def add_multiple_terms(self, terms):
        if type(terms) == list:
            for term in terms:
                self.add_term(term)
        else:
            raise TypeError("'terms' has to be a list of dictionaries")#

    def pad_with_identity(self, ttn, mode="safe"):
        """
        Pads all terms with an identity according to the reference TTN

        Parameters
        ----------
        ttn : TreeTensorNetwork
            TTN with reference to which the identities are to be padded. From
            here the site_ids and operator dimension is inferred.
        mode : string, optional
            Whether to perform checks ('safe') or not ('risky').
            For big TTN the checks can take a long time.
            The default is 'safe'.

        Returns
        -------
        None.

        """
        if mode == "safe":
            if not self.is_compatible_with(ttn):
                raise NotCompatibleException("Hamiltonian and TTN are incompatible")
        elif mode != "risky":
            raise ValueError(f"{mode} is not a valied option for 'mode'. (Only 'safe' and 'risky are)!")

        for site_id in ttn.nodes:

            site_node = ttn.nodes[site_id]
            physical_dim = prod(site_node.shape_of_legs(site_node.open_legs))
            site_identity = eye(physical_dim)

            for term in self.terms:
                if not (site_id in term):
                    term[site_id] = site_identity

    def is_compatible_with(self, ttn):
        """
        Checks if the Hamiltonian is compatible with the givent TTN.

        Parameters
        ----------
        ttn : TreeTensorNetwork
            The TTN to be checked against.

        Returns
        -------
        compatability: bool
            If the current Hamiltonian ist compatible with the given TTN.

        """

        for term in self.terms:
            for site_id in term:
                if not site_id in ttn.nodes:
                    return False

        return True

    def to_matrix(self, ttn):
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
        self.pad_with_identity(ttn)

        first = True
        for term in self.terms:

            tensor = term[ttn.root_id]
            tensor = self._to_matrix_rec(ttn, ttn.root_id, term, tensor)

            if first:
                matrix = tensor
            else:
                matrix += tensor

        return matrix

    def _to_matrix_rec(self, ttn, node_id, term, tensor):
        tensor = term[node_id]
        for child_id in ttn.nodes[node_id].children_legs:
            child_tensor = term[child_id]
            tensor = tensordot(tensor, child_tensor, axes=0)
            tensor = transpose(tensor, (0,2,1,3))
            tensor_half_dim = tensor.ndim / 2
            tensor = reshape(tensor, (tensor_half_dim, tensor_half_dim))

            tensor = self._to_matrix_rec(ttn, child_id, term, tensor)

        return tensor

    def __add__(self, other_hamiltonian):

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

def random_terms(num_of_terms, possible_operators, sites, min_strength = -1, max_strength = 1,
                 min_num_sites=2,  max_num_sites=2):
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

    rterms= []

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