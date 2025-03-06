"""
This module contains functions to generate Lindbladian superoperators to be
applied to vectorised density matrices.
"""
from typing import List, Dict, Union
from fractions import Fraction
from copy import copy

from numpy import ndarray, isreal, eye, allclose, all as np_all
from scipy.linalg import issymmetric, ishermitian

from .hamiltonian import Hamiltonian, deal_with_term_input
from .tensorproduct import TensorProduct

def generate_lindbladian(hamiltonian: Hamiltonian,
                         jump_operators: Union[List[tuple[Fraction, str, TensorProduct]], List[TensorProduct], TensorProduct, None],
                         jump_operator_dict: Dict[str, ndarray],
                         jump_coeff_mapping: Dict[str, complex],
                         ket_suffix: str = "_ket",
                         bra_suffix: str = "_bra"
                         ) -> Hamiltonian:
    """
    Generates a Linfbladian superoperator from a given system Hamiltonian.
    
    Args:
        hamiltonian (Hamiltonian): The system Hamiltonian.
        jump_operators (Union[List[tuple[Fraction, str, TensorProduct]],
            List[TensorProduct], TensorProduct, None]): The jump operators.
            Each operator is made up of a fractional and symbolic coefficient,
            and an operator in the form of a tensor product.
        jump_operator_dict (Dict[str, ndarray]): A dictionary mapping the
            symbolic jump operators to their numerical values.
        jump_coeff_mapping (Dict[str, complex]): A dictionary mapping the
            symbolic jump operator coefficients to their numerical values.
        ket_suffix (str, optional): The suffix for ket identifiers.
            Defaults to "_ket".
        bra_suffix (str, optional): The suffix for bra identifiers.
            Defaults to "_bra".
    
    Returns:
        Hamiltonian: The Lindbladian superoperator.

    """
    lindbladian = Hamiltonian(conversion_dictionary={})
    _add_hamiltonian_ket_terms(lindbladian, hamiltonian, ket_suffix)
    _add_hamiltonian_bra_terms(lindbladian, hamiltonian, bra_suffix)
    jump_operators = deal_with_term_input(jump_operators)
    _add_jump_operators(lindbladian, jump_operators, jump_operator_dict,
                        jump_coeff_mapping, ket_suffix, bra_suffix)
    _add_jump_operator_products(lindbladian,
                                jump_operators,
                                copy(jump_operator_dict),
                                ket_suffix,
                                bra_suffix)
    return lindbladian

def _add_hamiltonian_ket_terms(lindbladian: Hamiltonian,
                               hamiltonian: Hamiltonian,
                               ket_suffix: str) -> Hamiltonian:
    """
    Add the unchanged terms of the Hamiltonian to the Lindbladian.

    Args:
        lindbladian (Hamiltonian): The Lindbladian superoperator.
        hamiltonian (Hamiltonian): The system Hamiltonian.
        ket_suffix (str): The suffix for ket identifiers.
    
    Returns:
        Hamiltonian: The modified Lindbladian superoperator.

    """
    for term in hamiltonian.terms:
        new_tp = term[2].add_suffix(ket_suffix)
        lindbladian.add_term((term[0], term[1], new_tp))
    # We need to add the numerical values of the operaotr
    lindbladian.conversion_dictionary.update(hamiltonian.conversion_dictionary)
    lindbladian.coeffs_mapping.update(hamiltonian.coeffs_mapping)
    return lindbladian

def _add_hamiltonian_bra_terms(lindbladian: Hamiltonian,
                              hamiltonian: Hamiltonian,
                              bra_suffix: str) -> Hamiltonian:
    """
    Add the terms of the system Hamiltonian acting on the bra side to the
    Lindbladian.

    Args:
        lindbladian (Hamiltonian): The Lindbladian superoperator.
        hamiltonian (Hamiltonian): The system Hamiltonian.
        bra_suffix (str): The suffix for bra identifiers.

    Returns:
        Hamiltonian: The modified Lindbladian superoperator.

    """
    sym_dict = _find_symmetric_operators(hamiltonian.conversion_dictionary)
    for term in hamiltonian.terms:
        # Because this term is applied to the bra side, while the ket side
        # is trivial.
        new_tp = term[2].transpose(sym_dict)
        new_tp = new_tp.add_suffix(bra_suffix)
        # -1 comes from the commutator
        lindbladian.add_term((-1*term[0], term[1], new_tp))
    # The transposed operators need their numerical values.
    transpose_dict = {label + "_T": operator.T
                      for label, operator in hamiltonian.conversion_dictionary.items()
                      if not sym_dict[label]}
    lindbladian.conversion_dictionary.update(transpose_dict)
    return lindbladian

def _add_jump_operators(lindbladian: Hamiltonian,
                        jump_operators: List[tuple[Fraction, str, TensorProduct]],
                        jump_operator_dict: Dict[str, ndarray],
                        jump_coeff_mapping: Dict[str, complex],
                        ket_suffix: str,
                        bra_suffix: str
                        ) -> Hamiltonian:
    """
    Add the jump operators to the Lindbladian.

    Args:
        lindbladian (Hamiltonian): The Lindbladian superoperator.
        jump_operators (List[tuple[Fraction, str, TensorProduct]]): The jump
            operators.
        jump_operator_dict (Dict[str, ndarray]): A dictionary mapping the
            symbolic jump operators to their numerical values.
        jump_coeff_mapping (Dict[str, complex]): A dictionary mapping the
            symbolic jump operator coefficients to their numerical values.
        ket_suffix (str): The suffix for ket identifiers.
        bra_suffix (str): The suffix for bra identifiers.

    Returns:
        Hamiltonian: The modified Lindbladian superoperator.

    """
    real_dict = _find_real_operators(jump_operator_dict)
    for jump_operator in jump_operators:
        frac = jump_operator[0]
        coeff = jump_operator[1] + "*j"
        op = jump_operator[2]
        ket_tp = op.add_suffix(ket_suffix)
        bra_tp = op.add_suffix(bra_suffix)
        bra_tp = bra_tp.conjugate(real_dict=real_dict)
        full_tp = ket_tp.otimes(bra_tp, to_copy=True)
        lindbladian.add_term((frac, coeff, full_tp))
    # The conjugated operators need their numerical values.
    conjugate_dict = {label + "_conj": operator.conj()
                      for label, operator in jump_operator_dict.items()
                      if not real_dict[label]}
    lindbladian.conversion_dictionary.update(conjugate_dict)
    lindbladian.conversion_dictionary.update(jump_operator_dict)
    # Add we need to add the numerical values of the coefficients
    i_coeffs = {label + "*j": 1j*value
                for label, value in jump_coeff_mapping.items()}
    lindbladian.coeffs_mapping.update(i_coeffs)

def _add_jump_operator_products(lindbladian: Hamiltonian,
                                jump_operators: List[tuple[Fraction, str, TensorProduct]],
                                jump_operator_dict: Dict[str, ndarray],
                                ket_suffix: str,
                                bra_suffix: str
                                ) -> Hamiltonian:
    """
    Adds the terms to the Lindbladian that come from the products of jump
    operators.

    Args:
        lindbladian (Hamiltonian): The Lindbladian superoperator.
        jump_operators (List[tuple[Fraction, str, TensorProduct]]): The jump
            operators.
        jump_operator_dict (Dict[str, ndarray]): A dictionary mapping the
            symbolic jump operators to their numerical values.
        jump_coeff_mapping (Dict[str, complex]): A dictionary mapping the
            symbolic jump operator coefficients to their numerical values.
        ket_suffix (str): The suffix for ket identifiers.
        bra_suffix (str): The suffix for bra identifiers.

    Returns:
        Hamiltonian: The modified Lindbladian superoperator.

    """
    jop_conv_dict = copy(jump_operator_dict)
    id_dict = _find_identity_operators(jump_operator_dict)
    herm_dict = _find_hermitian_operators(jump_operator_dict)
    for op, op_value in jump_operator_dict.items():
        if not id_dict[op] and not herm_dict[op]:
            jop_conv_dict[op+"_H"] = op_value.conj().T
            id_dict[op+"_H"] = False
    for jump_operator in jump_operators:
        frac = -1 * jump_operator[0] / 2
        coeff = jump_operator[1] + "*j"
        op = jump_operator[2]
        op_adj = op.conjugate_transpose(herm_dict=herm_dict)
        op_mult = op_adj.multiply(op,
                                  identity_dict=id_dict,
                                  conversion_dict=jop_conv_dict)
        sym_dict = _find_symmetric_operators(jop_conv_dict)
        op_mult_transp = op_mult.transpose(sym_dict)
        new_frac = -1 * frac
        ket_tp = op_mult.add_suffix(ket_suffix)
        bra_tp = op_mult_transp.add_suffix(bra_suffix)
        lindbladian.add_term((frac, coeff, ket_tp))
        lindbladian.add_term((new_frac, coeff, bra_tp))
        lindbladian.conversion_dictionary.update(jop_conv_dict)
        transpose_dict = {label + "_T": operator.T
                        for label, operator in jop_conv_dict.items()
                        if not sym_dict[label]}
        lindbladian.conversion_dictionary.update(transpose_dict)
    return lindbladian

def _find_symmetric_operators(operator_dict: Dict[str,ndarray]
                              ) -> Dict[str,bool]:
    """
    Dictionary mapping operators to whether they are symmetric.

    Args:
        operator_dict (Dict[str,ndarray]): A dictionary mapping operators to
            their numerical values.
    
    Returns:
        Dict[str,bool]: A dictionary mapping operators to whether they are
            symmetric.

    """
    return {op_label: issymmetric(op, rtol=1e-05, atol=1e-10)
            for op_label, op in operator_dict.items()}

def _find_real_operators(operator_dict: Dict[str,ndarray]
                         ) -> Dict[str,bool]:
    """
    Dictionary mapping operators to whether they are real.

    Args:
        operator_dict (Dict[str,ndarray]): A dictionary mapping operators to
            their numerical values.
    
    Returns:
        Dict[str,bool]: A dictionary mapping operators to whether they are
            real.

    """
    return {op_label: np_all(isreal(op))
            for op_label, op in operator_dict.items()}

def _find_hermitian_operators(operator_dict: Dict[str,ndarray]
                              ) -> Dict[str,bool]:
    """
    Dictionary mapping operators to whether they are Hermitian.

    Args:
        operator_dict (Dict[str,ndarray]): A dictionary mapping operators to
            their numerical values.
    
    Returns:
        Dict[str,bool]: A dictionary mapping operators to whether they are
            Hermitian.

    """
    return {op_label: ishermitian(op, rtol=1e-05, atol=1e-10)
            for op_label, op in operator_dict.items()}


def _find_identity_operators(operator_dict: Dict[str,ndarray]
                             ) -> Dict[str,bool]:
    """
    Dictionary mapping operators to whether they are the identity.

    Args:
        operator_dict (Dict[str,ndarray]): A dictionary mapping operators to
            their numerical values.

    Returns:
        Dict[str,bool]: A dictionary mapping operators to whether they are the
            identity.
    
    """
    id_dict = {}
    for op_label, op_value in operator_dict.items():
        dim = op_value.shape[0]
        identity = eye(dim)
        id_dict[op_label] = allclose(op_value, identity)
    return id_dict
