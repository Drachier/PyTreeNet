from numpy.random import default_rng
from numpy import prod, eye, tensordot, reshape, transpose, kron

from .ttn_exceptions import NotCompatibleException

class Hamiltonian(object):
    """
    Represents the Hamiltonian on a TTN.
    The entries of the main list should represent the terms of the Hamiltonian
    and be a dictionary. The key is an identifier which could be matched to a
    TensorNode in the TTN and the value is an operator that is to be applied to
    that node/site.
    """

    def __init__(self, terms=None, conversion_dictionary=None):
        """
        Parameters
        ----------
        terms : list of dictionaries, optional
            A list of dictionaries containing the terms of the Hamiltonian. The
            keys are identifiers of the site to which the value, an operator,
            is to be applied. (Operators can be symbolic, i.e. strings or explicit
            i.e. ndarrays)
            The default is None.
        conversion_dictionary : dict
            A dictionary that contains keys corresponding to certain tensors.
            Thus the terms can contain labels rather than the whole numpy arrays
            representing the operators.

        """
        if terms == None:
            self.terms = []
        else:
            self.terms = terms

        if conversion_dictionary == None:
            self.conversion_dictionary = []
        else:
            self.conversion_dictionary = conversion_dictionary

        self.check_conversion_dict_valid()

    def check_conversion_dict_valid(self):
        for label in self.conversion_dictionary:
            operator = self.conversion_dictionary[label]

            shape = operator.shape
            assert len(shape) == 2,  f"Operator with label {label} is not a matrix!"
            assert shape[0] == shape[1], f"Matrix with label {label} is not square!"

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

    def pad_with_identity(self, reference_ttn, mode="safe", identity=None):
        """
        Pads all terms with an identity according to the reference reference_ttn

        Parameters
        ----------
        reference_ttn : TreeTensorNetwork
            reference_ttn with reference to which the identities are to be padded. From
            here the site_ids and operator dimension is inferred.
        mode : string, optional
            Whether to perform checks ('safe') or not ('risky').
            For big reference_ttn the checks can take a long time.
            The default is 'safe'.
        identity :
            If None, an appropriately sized identity is determined. Else the
            value is inserted as a placeholder.

        Returns
        -------
        None.

        """
        if mode == "safe":
            if not self.is_compatible_with(reference_ttn):
                raise NotCompatibleException("Hamiltonian and reference_ttn are incompatible")
        elif mode != "risky":
            raise ValueError(f"{mode} is not a valied option for 'mode'. (Only 'safe' and 'risky are)!")

        for site_id in reference_ttn.nodes:

            if identity == None:
                site_node = reference_ttn.nodes[site_id]
                physical_dim = prod(site_node.shape_of_legs(site_node.open_legs))
                site_identity = eye(physical_dim)
            else:
                site_identity = identity

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

    def to_tensor(self, ref_ttn, use_padding=False):
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
        permutation = list(range(0,full_tensor.ndim,2))
        permutation.extend(list(range(1,full_tensor.ndim,2)))
        full_tensor = full_tensor.transpose(permutation)
        return full_tensor

    def _to_tensor_rec(self, ttn, node_id, term, tensor):
        for child_id in ttn.nodes[node_id].children_legs:
            child_tensor = self.conversion_dictionary[term[child_id]]
            tensor = tensordot(tensor, child_tensor, axes=0)
            tensor = self._to_tensor_rec(ttn, child_id, term, tensor)

        return tensor

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
        matrix = self.to_tensor(ttn)
        half_dim = matrix.ndim / 2
        matrix_size = prod(matrix.shape[0:half_dim])

        return matrix.reshape((matrix_size, matrix_size))

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