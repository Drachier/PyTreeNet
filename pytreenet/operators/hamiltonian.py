"""
This module provides a class to represent the Hamiltonian of a system.

The Hamiltonian of a quantum mechanical system can be interpreted as the
a representation of the energy or potential of the system. It is usually used
to define a model in the first place. For quantum systems made up of multiple
smaller systems the Hamiltonian is defined as a sum over tensor products.
These products define one operator per subsystem and fully define the
Hamiltonian.

.. math::
    H = sum_i \\bigotimes_j A_{i}^[j]
where :math:`A_{i}^{[j]}` is the operator acting on the j-th subsystem of the
as part of the i-th term of the Hamiltonian.
"""
from __future__ import annotations
from typing import Dict, Union, List
from enum import Enum, auto
from numpy import asarray, ndarray
from fractions import Fraction

from .operator import NumericOperator
from .tensorproduct import TensorProduct
from ..core.ttn import TreeTensorNetwork
from ..core.tree_structure import TreeStructure
from ..util.std_utils import compare_lists_by_value
from ..util.ttn_exceptions import NotCompatibleException

class PadMode(Enum):
    """
    When padding a Hamiltonian, we can decide to check its compatabilty with a
    given tree tensor network or not.
    """
    risky = auto()
    safe = auto()

class Hamiltonian():
    """
    Represents the Hamiltonian on a TTN.

    The entries of the main list should represent the terms of the Hamiltonian
    and be a dictionary. The key is an identifier which could be matched to a
    TensorNode in the TTN and the value is an operator that is to be applied to
    that node/site.

    Attributes:
        terms (List[TensorProduct]): A list of tensor products, each
            representing one term of the Hamiltonian. Summin over the list
            would theoretically yield the total Hamiltonian.
        conversion_dictionary (Dict[str,ndarray]): A dictionary that can be
            used to convert symbolically given terms of the Hamiltonian into
            actual numeric arrays.
    """

    def __init__(self, terms: Union[List[tuple[Fraction, str, TensorProduct]], List[TensorProduct], TensorProduct, None] = None,
                 conversion_dictionary: Union[Dict[str, ndarray],None] = None,
                 coeffs_mapping: Union[Dict[str,complex],None] = None):
        """
        Initialises a Hamiltonian from a number of terms represented by a TensorProduct each:
            H = sum( terms )

        Args:
            terms (List[Tuple[Fraction, str, Tensorproduct]], List[Tensorproduct], Tensorproduct, optional): A list of TensorProduct making up the
                Hamiltonian. Defaults to None.
            conversion_dictionary (dict, optional): A conversion dictionary might be supplied.
                It is used, if the tensor products are symbolic. Defaults to None.
        """
        if terms is None:
            self.terms = []
        elif isinstance(terms, TensorProduct):
            self.terms = [(Fraction(1),"1",terms)]
        else:
            if all([isinstance(term, TensorProduct) for term in terms]):
                self.terms = [(Fraction(1),"1",term) for term in terms]
            else:
                self.terms = terms


        if coeffs_mapping is None:
            coeffs_mapping = {"1" : 1}

        if conversion_dictionary is None:
            self.conversion_dictionary = {}
        else:
            self.conversion_dictionary = conversion_dictionary

        self.coeffs_mapping = coeffs_mapping

        
    def __str__(self) -> str:
        """
        Returns a string representation of the Hamiltonian.
        """
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

    def add_term(self, term: Union[TensorProduct, tuple[Fraction, str, TensorProduct]]):
        """
        Adds a term to the Hamiltonian.

        Args:
            term (TensorProduct): The term to be added in the form of a TensorProduct
        """
        if isinstance(term, tuple):
            self.terms.append(term)
        else:
            self.terms.append((Fraction(1),"1",term))
        

    def add_hamiltonian(self, other: Hamiltonian):
        """
        Adds one Hamiltonian to this Hamiltonian. The other Hamiltonian will not be modified.

        Args:
            other (Hamiltonian): Hamiltonian to be added.
        """
        self.terms.extend(other.terms)
        self.conversion_dictionary.update(other.conversion_dictionary)
        self.coeffs_mapping.update(other.coeffs_mapping)

    def add_multiple_terms(self, terms: Union[list[TensorProduct], list[tuple[Fraction, str, TensorProduct]]]):
        """
        Add multiple terms to this Hamiltonian

        Args:
            terms (list[TensorProduct]): Terms to be added.
        """
        if all([isinstance(term, TensorProduct) for term in terms]):
            self.terms.extend([(Fraction(1),"1",term) for term in terms])
        else:
            self.terms.extend(terms)

    def is_compatible_with(self, ttn: TreeTensorNetwork) -> bool:
        """
        Returns wether the Hamiltonian is compatible with the provided TTN.
        
        Compatibility means that all node identifiers that appear any term of
        this Hamiltonian are identifiers of nodes in the TTN.

        Args:
            ttn (TreeTensorNetwork): The TTN to check against.

        Returns:
            bool: Whether the two are compatible or not.
        """
        for _,_,term in self.terms:
            for site_id in term:
                if not site_id in ttn.nodes:
                    return False
        return True

    def perform_compatibility_checks(self, mode: PadMode,
                                     reference_ttn: TreeTensorNetwork):
        """
        Performs the check of the mode and the check of compatibility, if desired.

        Args:
            mode (PadMode, optional): 'safe' performs a compatability check
                with the reference ttn. Risky will not run this check, which
                might be time consuming for large TTN. Defaults to
                PadMode.safe.

        Raises:
            NotCompatibleException: If the Hamiltonian and TTN are not compatible
        """
        if mode == PadMode.safe:
            if not self.is_compatible_with(reference_ttn):
                errstr = "Hamiltonian and reference_ttn are incompatible!"
                raise NotCompatibleException(errstr)

    def pad_with_identities(self, reference_ttn: TreeTensorNetwork,
                          mode: PadMode = PadMode.safe, 
                          symbolic: bool = True) -> Hamiltonian:
        """
        Pads a Hamiltonian with identities.

        Returns a Hamiltonian, where all terms are padded with an identity
        according to the given reference tree tensor network.
        Make sure to update the resulting Hamiltonian's conversion dictionary
        to include any new identity.

        Args:
            reference_ttn (TreeTensorNetwork): Provides the structure on which
                padding should occur. Furthermore the dimension of the open
                legs of each provide the new identities' dimensions.
            mode (PadMode, optional): 'safe' performs a compatability check
                with the reference TTN. Risky will not run this check, which
                might be time consuming for large TTN. Defaults to 
                PadMode.safe.
            symbolic (bool, optional): Whether the terms should be padded with
                a symbolic identity or an actual array. Defaults to True.

        Returns:
            Hamiltonian: A new Hamiltonian with all terms being padded.
        """
        self.perform_compatibility_checks(mode=mode, reference_ttn=reference_ttn)
        new_terms = []
        for frac, coeff, term in self.terms:
            new_term = term.pad_with_identities(reference_ttn, symbolic=symbolic)
            new_terms.append((frac, coeff, new_term))
        return Hamiltonian(new_terms, conversion_dictionary=self.conversion_dictionary, coeffs_mapping=self.coeffs_mapping)

    def to_matrix(self, ref_ttn: TreeTensorNetwork, use_padding: bool = True,
                  mode: PadMode = PadMode.safe) -> NumericOperator:
        """
        Creates a numeric operator that is equivalent to the Hamiltonian.

        The resulting operator can get very large very fast, so this should
        only be used for debugging. The result is a matrix valued operator.
        The resulting operator is a matrix where the dimensions are ordered
        according to the open dimensions of the nodes in the reference TTN
        in the order the nodes identifiers are saved in the TTN.

        Args:
            ref_ttn (TreeTensorNetwork): TTN giving the tree structure which
                the Hamiltonian should respect.
            use_padding (bool, optional): Enable, if the Hamiltonian requires
                padding with respect to the reference TTN. Defaults to True.
            mode (PadMode, optional): 'safe' performs a compatability check
                with the reference TTN. Risky will not run this check, which
                might be time consuming for large TTN. Defaults to
                PadMode.safe.

        Returns:
            NumericOperator: Operator corresponding to the Hamiltonian. The
                actual array is matrix valued.
        """
        self.perform_compatibility_checks(mode=mode, reference_ttn=ref_ttn)
        if use_padding:
            self.pad_with_identities(ref_ttn)
        full_tensor = asarray([0], dtype=complex)
        identifiers = list(ref_ttn.nodes.keys())
        for i, (frac, coeff, term) in enumerate(self.terms):
            term_operator = term.into_operator(conversion_dict=self.conversion_dictionary,
                                               order=identifiers)
            if i == 0:
                full_tensor = term_operator.operator * float(frac) * self.coeffs_mapping[coeff]
            else:
                full_tensor = term_operator.operator * float(frac) * self.coeffs_mapping[coeff] + full_tensor
        return NumericOperator(full_tensor.T, identifiers)

    def to_tensor(self, ref_ttn: TreeTensorNetwork, use_padding: bool = True,
                  mode: PadMode = PadMode.safe) -> NumericOperator:
        """
        Creates a NumericOperator that is equivalent to the Hamiltonian.

        The resulting operator can get very large very fast, so this should
        only be used for debugging. The result is a tensor with multiple legs.
        The legs are ordered according to the open dimensions of the nodes in
        the reference TTN in the order the nodes identifiers are saved in the
        TTN.

        Args:
            ref_ttn (TreeTensorNetwork): TTN giving the tree structure which
                the Hamiltonian should respect.
            use_padding (bool, optional): Enable, if the Hamiltonian requires
                padding with respect to the reference TTN. Defaults to True.
            mode (PadMode, optional): 'safe' performs a compatability check
                with the reference TTN. Risky will not run this check, which
                might be time consuming for large TTN. Defaults to PadMode.safe.

        Returns:
            NumericOperator: Operator corresponding to the Hamiltonian. The
                actual array is tensor valued.
        """
        matrix_operator = self.to_matrix(ref_ttn,use_padding=use_padding,mode=mode)
        shape = [node.open_dimension() for node in ref_ttn.nodes.values()]
        # remove 0 indices
        #shape = [dim for dim in shape if dim != 0]
        shape *= 2
        tensor_operator = matrix_operator.operator.reshape(shape)
        return NumericOperator(tensor_operator, matrix_operator.node_identifiers)

    def contains_duplicates(self) -> bool:
        """
        Checks, if there are duplicates of terms.
        
        Can be especially important after padding.

        Returns:
            bool: True if there are duplicates, False otherwise
        """
        terms_comp = [term[2] for term in self.terms]
        dup = [term for term in terms_comp if terms_comp.count(term) > 1]
        return len(dup) > 0
