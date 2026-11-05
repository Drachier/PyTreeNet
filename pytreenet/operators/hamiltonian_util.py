"""
Tools that act on a Hamiltonian and on the operators of its tensor products.

The construction of a state diagram eliminates over operator symbols, see Cakir et al.,
2025 (arXiv:2502.18630, https://doi.org/10.48550/arXiv.2502.18630) and, for the bipartite
part, Ren et al., 2020 (arXiv:2006.02056, https://doi.org/10.48550/arXiv.2006.02056). Two
symbols naming linearly dependent matrices are therefore two independent directions to
it, and the bond it builds is above the operator-Schmidt rank of the operator. The
condition that the operators on each node be linearly independent is not part of either
work. It is the additional assumption under which the symbolic objective coincides with
the operator-Schmidt rank, and :func:`rewrite_over_independent_operators` establishes it.
The smallest case is a controlled gate whose target carries a multiple of the identity
beside the identity itself.

Establishing the condition does not make the construction attain the bound. The rewrite
never raises the rank of the coefficient matrix and drives it down to the
operator-Schmidt rank, but what the construction attains is that matrix's rank with the
coefficient symbols read as independent indeterminates, and the rewrite does not control
that. By giving a combined coefficient a symbol of its own it can raise it, and on a
three-qubit Lindbladian on a chain of six nodes the bond on the middle cut consequently
went from five, the operator-Schmidt rank, to six. This is why
:class:`pytreenet.ttno.state_diagram.OperatorRewrite` offers to build both presentations
and keep the cheaper one.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray

from .hamiltonian import ONE_SYMBOL, Hamiltonian
from .tensorproduct import TensorProduct

__all__ = ["rewrite_over_independent_operators"]

_NEW_COEFF_PREFIX = "indep_coeff_"


class _Exact:
    """A complex number with rational real and imaginary parts."""

    __slots__ = ("im", "re")

    def __init__(self, re: Fraction, im: Fraction | None = None) -> None:
        """
        Store the real and imaginary parts.

        Args:
            re (Fraction): The real part.
            im (Fraction | None): The imaginary part, zero when omitted.
        """
        self.re = re
        self.im = Fraction(0) if im is None else im

    @classmethod
    def of(cls, value: complex | _Exact) -> _Exact:
        """
        Return ``value`` as an exact number, a float converting without rounding.

        Args:
            value (complex | _Exact): The number to convert.

        Returns:
            _Exact: The exact number.
        """
        if isinstance(value, _Exact):
            return value
        if isinstance(value, Fraction):
            return cls(value)
        number = complex(value)
        return cls(Fraction(number.real), Fraction(number.imag))

    def __add__(self, other: _Exact) -> _Exact:
        """Return the exact sum."""
        return _Exact(self.re + other.re, self.im + other.im)

    def __sub__(self, other: _Exact) -> _Exact:
        """Return the exact difference."""
        return _Exact(self.re - other.re, self.im - other.im)

    def __mul__(self, other: _Exact) -> _Exact:
        """Return the exact product."""
        return _Exact(self.re * other.re - self.im * other.im,
                      self.re * other.im + self.im * other.re)

    def inverse(self) -> _Exact:
        """
        Return the exact reciprocal of a nonzero number.

        Returns:
            _Exact: One divided by this number.
        """
        norm = self.re * self.re + self.im * self.im
        return _Exact(self.re / norm, -self.im / norm)

    def is_real(self) -> bool:
        """Return whether the imaginary part is exactly zero."""
        return not self.im

    def __bool__(self) -> bool:
        """Return whether the number is nonzero, exactly and with no tolerance."""
        return bool(self.re) or bool(self.im)

    def __complex__(self) -> complex:
        """Return the nearest double precision number."""
        return complex(float(self.re), float(self.im))

    def __repr__(self) -> str:
        """Return a readable form."""
        return f"_Exact({self.re}, {self.im})"


_ZERO = _Exact(Fraction(0))
_ONE = _Exact(Fraction(1))


def _reduced_row_echelon(
    matrix: List[List[_Exact]]
) -> Tuple[List[List[_Exact]], List[int]]:
    """
    Return the nonzero rows of the reduced row echelon form and the pivot columns.

    The pivot of a column is its first nonzero entry, with no comparison of magnitudes,
    which is sound because the arithmetic is exact.

    Args:
        matrix (List[List[_Exact]]): Rows of exact numbers, all of one length.

    Returns:
        Tuple[List[List[_Exact]], List[int]]: The reduced rows, each with a leading one,
            and the column of each leading one.
    """
    rows = [list(row) for row in matrix]
    num_cols = len(rows[0]) if rows else 0
    pivots: List[int] = []
    rank = 0
    for col in range(num_cols):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col].inverse()
        rows[rank] = [entry * scale for entry in rows[rank]]
        for i, row in enumerate(rows):
            if i != rank and row[col]:
                factor = row[col]
                rows[i] = [entry - factor * lead
                           for entry, lead in zip(row, rows[rank])]
        pivots.append(col)
        rank += 1
        if rank == len(rows):
            break
    return rows[:rank], pivots


def _reconstructs(operator: ndarray,
                  basis: List[ndarray],
                  coordinates: Dict[int, _Exact],
                  tol: float
                  ) -> bool:
    """
    Return whether the exact coordinates rebuild the operator in double precision too.

    A dependence that holds over the rationals can still be useless. When the operators
    are nearly but not exactly dependent, which is what float data such as a Lindbladian
    gives, the basis is nearly rank deficient, the exact coordinates run to ``1e18`` and
    beyond, and evaluating them in double precision loses the operator entirely. The
    caller then keeps the operator as its own direction, which costs a bond of the TTNO
    and never the operator it represents.

    Args:
        operator (ndarray): The operator the coordinates are meant to rebuild.
        basis (List[ndarray]): The independent operators found so far.
        coordinates (Dict[int, _Exact]): The exact coordinates on them.
        tol (float): The relative size the rebuilt operator may differ by.

    Returns:
        bool: True when the coordinates rebuild the operator to ``tol`` of its own size.
    """
    rebuilt = np.zeros_like(np.asarray(operator), dtype=complex)
    for position, value in coordinates.items():
        rebuilt = rebuilt + complex(value) * np.asarray(basis[position])
    scale = float(np.abs(operator).max())
    return bool(np.abs(rebuilt - operator).max() <= tol * max(scale, 1.0))


def _independent_subset(
    operators: List[ndarray], tol: float
) -> Tuple[List[int], List[Dict[int, _Exact]]]:
    """
    Return which operators form an independent set, and each one's exact coordinates.

    Args:
        operators (List[ndarray]): The distinct operators of one node.
        tol (float): The relative size a dependence may miss the operator by in double
            precision before it is refused, see :func:`_reconstructs`.

    Returns:
        Tuple[List[int], List[Dict[int, _Exact]]]: The indices of the independent
            operators, and for every operator its coordinates on them, keyed by the
            position within the independent set.
    """
    keep: List[int] = []
    vectors: List[List[_Exact]] = []
    coordinates: List[Dict[int, _Exact]] = []
    for index, operator in enumerate(operators):
        vector = [_Exact.of(entry) for entry in np.asarray(operator).ravel()]
        if vectors:
            # Solve vector = sum_b a_b vectors[b] exactly by reducing [vectors^T | vector].
            system = [[*(held[i] for held in vectors), vector[i]]
                      for i in range(len(vector))]
            reduced, pivots = _reduced_row_echelon(system)
            if len(vectors) not in pivots:  # consistent, so the operator is dependent
                coords = {pivot: row[-1]
                          for pivot, row in zip(pivots, reduced) if row[-1]}
                basis = [operators[i] for i in keep]
                if _reconstructs(operator, basis, coords, tol):
                    coordinates.append(coords)
                    continue
                # the dependence holds over the rationals but not in double precision,
                # so the operator is kept as its own direction instead
        coordinates.append({len(keep): _ONE})
        keep.append(index)
        vectors.append(vector)
    return keep, coordinates


def _node_rewrites(
    hamiltonian: Hamiltonian, tol: float
) -> Dict[str, Dict[str, Dict[str, _Exact]]]:
    """
    Return, for every node that carries a dependence, how to rewrite its operators.

    A node is skipped when some term does not name it, when one of its symbols has no
    matrix, when its matrices are not all of one shape, or when its operators are already
    independent, so a Hamiltonian the elimination already handles optimally is untouched.

    Args:
        hamiltonian (Hamiltonian): The Hamiltonian to inspect.
        tol (float): The relative size below which a number counts as noise, see
            :func:`rewrite_over_independent_operators`.

    Returns:
        Dict[str, Dict[str, Dict[str, _Exact]]]: ``{node: {symbol: {kept symbol:
            coefficient}}}`` for each node whose operators are dependent.
    """
    terms = hamiltonian.terms
    conversion = hamiltonian.conversion_dictionary
    per_node: Dict[str, List[str]] = {}
    for _, _, tensor_product in terms:
        for node_id, symbol in tensor_product.items():
            seen = per_node.setdefault(node_id, [])
            if symbol not in seen:
                seen.append(symbol)

    rewrites: Dict[str, Dict[str, Dict[str, _Exact]]] = {}
    for node_id, symbols in per_node.items():
        if any(node_id not in tp for _, _, tp in terms):
            continue  # an absent node is an implicit identity this pass cannot name
        if len(symbols) < 2 or any(s not in conversion for s in symbols):
            continue
        matrices = [np.asarray(conversion[s]) for s in symbols]
        if len({m.shape for m in matrices}) != 1:
            continue
        keep, coordinates = _independent_subset(matrices, tol)
        if len(keep) == len(symbols):
            continue  # already independent, which is the condition the elimination needs
        # The exact solve reads a nearly dependent set as independent, and then the
        # coordinates on it are not determined to any useful precision. Require the kept
        # operators to be a basis in double precision too, or leave the node alone.
        stacked = np.array([np.asarray(matrices[i]).ravel() for i in keep])
        values = np.linalg.svd(stacked, compute_uv=False)
        if int(np.sum(values > tol * values[0])) != len(keep):
            continue
        kept_symbols = [symbols[i] for i in keep]
        rewrites[node_id] = {
            symbol: {kept_symbols[position]: value for position, value in coords.items()}
            for symbol, coords in zip(symbols, coordinates)
        }
    return rewrites


def rewrite_over_independent_operators(hamiltonian: Hamiltonian,
                                       tol: float = 1e-10
                                       ) -> Hamiltonian:
    """
    Return an equal Hamiltonian whose operators are independent on every node.

    On each node the operators the terms name are written over a subset of them that is
    linearly independent, and the terms are expanded and collected accordingly. Pass a
    Hamiltonian that names every node of the reference tree, since a node some term does
    not name is left alone, see
    :meth:`pytreenet.operators.hamiltonian.Hamiltonian.pad_with_identities`. A node is
    also left alone when it carries fewer than two operators, when one of its symbols has
    no matrix, when its matrices differ in shape, when its operators are already
    independent, or when their dependence does not hold in double precision.

    The arithmetic is exact and the decision is not, and both are deliberate.
    Coefficients are combined as Gaussian rationals and a float converts to a rational
    without rounding, so the returned Hamiltonian represents the same operator and no
    rounding decides which terms survive. Whether two operators count as linearly
    dependent is a numerical judgement, because an exactly rational dependence between
    float matrices can be an artefact of their stored digits, and ``tol`` is where that
    judgement is made.

    No operator is created. Every tensor product returned names, on each node, one of the
    symbols that node already carried, so the conversion dictionary is carried over
    unchanged. Combining coefficients can turn one complex, and a complex coefficient
    needs a symbol, so the coefficient mapping grows by one symbol per distinct complex
    value.

    Args:
        hamiltonian (Hamiltonian): The Hamiltonian to rewrite.
        tol (float, optional): The relative size below which a number counts as noise. A
            singular value below ``tol`` of the largest is not a direction, and a rebuilt
            operator differing by less than ``tol`` of the operator's largest entry is
            not an error. Defaults to 1e-10.

    Returns:
        Hamiltonian: An equal Hamiltonian, independent on every node that could be
            rewritten. The Hamiltonian passed in, and not a copy, when no node had to be
            rewritten, which a caller may test with ``is``. An empty Hamiltonian when
            every coefficient cancelled.
    """
    rewrites = _node_rewrites(hamiltonian, tol)
    if not rewrites:
        return hamiltonian

    coeffs = hamiltonian.coeffs_mapping
    # (symbol per node) -> exact coefficient, so that strings which become equal combine
    collected: Dict[Tuple[Tuple[str, str], ...], _Exact] = {}
    for frac, symbol, tensor_product in hamiltonian.terms:
        base = _Exact.of(coeffs.get(symbol, 1)) * _Exact.of(Fraction(frac))
        choices = []
        for node_id, operator in tensor_product.items():
            table = rewrites.get(node_id)
            if table is None or operator not in table:
                choices.append([((node_id, operator), _ONE)])
            else:
                choices.append([((node_id, kept), value)
                                for kept, value in table[operator].items()])
        for combination in product(*choices):
            value = base
            for _, weight in combination:
                if weight is not _ONE:
                    value = value * weight
            if not value:
                continue
            key = tuple(sorted(pair for pair, _ in combination))
            collected[key] = collected[key] + value if key in collected else value

    new_terms = []
    new_coeffs_mapping: Dict[str, complex] = {}
    coeff_symbol_of_value: Dict[Tuple[Fraction, Fraction], str] = {}
    for key, value in sorted(collected.items()):
        if not value:
            continue
        tensor_product = TensorProduct(dict(key))
        if value.is_real():
            new_terms.append((Fraction(value.re), ONE_SYMBOL, tensor_product))
            continue
        tag = (value.re, value.im)
        if tag not in coeff_symbol_of_value:
            name = f"{_NEW_COEFF_PREFIX}{len(coeff_symbol_of_value)}"
            while name in coeffs or name in new_coeffs_mapping:
                name = f"{name}_"
            coeff_symbol_of_value[tag] = name
            new_coeffs_mapping[name] = complex(value)
        new_terms.append((Fraction(1), coeff_symbol_of_value[tag], tensor_product))

    out = Hamiltonian(new_terms,
                      conversion_dictionary=hamiltonian.conversion_dictionary.copy(),
                      coeffs_mapping=coeffs.copy())
    out.update_mappings(coeffs_mapping=new_coeffs_mapping)
    return out
