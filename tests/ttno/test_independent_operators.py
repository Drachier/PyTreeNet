"""
Tests for rewriting a Hamiltonian over operators independent on every node.

The symbolic elimination reaches the operator-Schmidt rank only once the operators it
eliminates over are independent on each node, and it cannot establish that itself because
it holds them as symbols. These tests pin that the rewrite establishes it, that it leaves
the operator unchanged, and that a Hamiltonian already satisfying the condition is
returned untouched so the TTNO built from it does not move.
"""
import unittest
from fractions import Fraction

import numpy as np

from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.operators.lindbladian import generate_lindbladian
from pytreenet.random.random_special_ttns import random_mps
from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.special_ttn.pauli import (
    pauli_dense_lindbladian, pauli_lindbladian_ttno, pauli_product_mps, pauli_site_ids)
from pytreenet.ttno.independent_operators import rewrite_over_independent_operators
from pytreenet.ttno.state_diagram import StateDiagram, TTNOFinder
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator as TTNO

IDENTITY = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.diag([1.0, -1.0]).astype(complex)
PROJ_0 = np.diag([1.0, 0.0]).astype(complex)
PROJ_1 = np.diag([0.0, 1.0]).astype(complex)
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)


def chain_reference(num_sites: int):
    """Returns an MPS to use as the reference tree, and its site identifiers in order."""
    # rooted at the first site, so every node carries its parent leg first and the bond
    # to the left neighbour is leg 0 of the node on the right of the cut
    mps = random_mps(num_sites, 2, 2, root_site=0)
    site_ids = [f"site{i}" for i in range(num_sites)]
    assert set(site_ids) == set(mps.nodes), "the MPS does not use the expected site ids"
    return mps, site_ids


def controlled_gate_hamiltonian(site_ids, controls, target, target_matrix):
    """
    The three strings of a controlled gate, as MultiControlledGate builds them.

    Args:
        site_ids (list[str]): The site identifiers of the chain.
        controls (dict[int, int]): Control position mapped to the value it reads.
        target (int): The target position.
        target_matrix (np.ndarray): The one-qubit target operator.

    Returns:
        Hamiltonian: The gate as ``I + Pi_b (x) U - Pi_b (x) I``.
    """
    controlled, subtracted, identity_string = (TensorProduct() for _ in range(3))
    for position, value in controls.items():
        controlled.add_operator(site_ids[position], f"Proj{value}")
        subtracted.add_operator(site_ids[position], f"Proj{value}")
        identity_string.add_operator(site_ids[position], "I2")
    controlled.add_operator(site_ids[target], "U")
    subtracted.add_operator(site_ids[target], "I2")
    identity_string.add_operator(site_ids[target], "I2")
    hamiltonian = Hamiltonian()
    hamiltonian.add_term((Fraction(1), "1", controlled))
    hamiltonian.add_term((Fraction(-1), "1", subtracted))
    hamiltonian.add_term((Fraction(1), "1", identity_string))
    hamiltonian.update_mappings({"I2": IDENTITY,
                                 "Proj0": PROJ_0,
                                 "Proj1": PROJ_1,
                                 "U": target_matrix},
                                {"1": 1.0 + 0.0j})
    return hamiltonian


def chain_bonds(ttno, site_ids):
    """Returns the bond dimension on every cut of a chain TTNO, left to right."""
    return [ttno.tensors[site_ids[i + 1]].shape[0] for i in range(len(site_ids) - 1)]


def dense_matrix(ttno, site_ids):
    """Returns the dense operator of a TTNO with ``site_ids[0]`` the most significant leg."""
    matrix, order = ttno.as_matrix(order=list(site_ids))
    assert order == list(site_ids), f"as_matrix returned the order {order}"
    return matrix


def operator_schmidt_ranks(matrix, num_sites, tol=1e-10):
    """Returns the operator-Schmidt rank of a dense operator on every cut of a chain."""
    tensor = np.asarray(matrix).reshape([2] * (2 * num_sites))
    ranks = []
    for cut in range(1, num_sites):
        permutation = [*range(cut),
                       *range(num_sites, num_sites + cut),
                       *range(cut, num_sites),
                       *range(num_sites + cut, 2 * num_sites)]
        reshuffled = tensor.transpose(permutation).reshape(4 ** cut, -1)
        singular = np.linalg.svd(reshuffled, compute_uv=False)
        ranks.append(int((singular > tol).sum()))
    return ranks


def random_ket(rng):
    """Returns a normalised random qubit state vector."""
    vec = rng.normal(size=2) + 1j * rng.normal(size=2)
    return vec / np.linalg.norm(vec)


def random_hermitian(rng, dim):
    """Returns a random Hermitian ``dim x dim`` matrix."""
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (mat + mat.conj().T) / 2


def lindbladian_model(rng, num_qubits):
    """
    Returns the bond terms, site terms and jump operators of a random Lindbladian.

    Every nearest-neighbour pair carries a two-qubit term, qubits 0 and 2 carry a
    long-range one, every site a one-qubit term, and every site a lowering jump, with
    qubit 0 also dephasing.
    """
    bonds = {(q, q + 1): random_hermitian(rng, 4) for q in range(num_qubits - 1)}
    if num_qubits >= 3:
        bonds[(0, 2)] = random_hermitian(rng, 4)
    sites = {q: random_hermitian(rng, 2) for q in range(num_qubits)}
    jumps = {q: [(SIGMA_MINUS, 0.1 + 0.05 * q)] for q in range(num_qubits)}
    jumps[0].append((PAULI_Z, 0.07))
    return bonds, sites, jumps


def ttno_dense(ttno, site_ids):
    """The TTNO as a ``4^N x 4^N`` matrix in qubit order."""
    mat, order = ttno.as_matrix()
    dims = [int(round(np.sqrt(ttno.nodes[n].open_dimension()))) for n in order]
    tensor = mat.reshape(dims + dims)
    n = len(order)
    keep = [order.index(s) for s in site_ids]
    drop = [i for i in range(n) if i not in keep]
    tensor = tensor.transpose(keep + drop + [n + i for i in keep] + [n + i for i in drop])
    k = len(keep)
    return tensor.reshape(4 ** k, 4 ** k)


class TestLeavesIndependentOperatorsAlone(unittest.TestCase):
    """A Hamiltonian the elimination already handles optimally must not move."""

    def test_pauli_sum_is_returned_unchanged(self):
        """Pauli matrices are independent, so there is nothing to rewrite."""
        mps, site_ids = chain_reference(4)
        hamiltonian = Hamiltonian()
        for operator in ("X", "Y", "Z"):
            term = TensorProduct({site_ids[0]: operator, site_ids[2]: operator})
            hamiltonian.add_term((Fraction(1), "1", term))
        hamiltonian.update_mappings({"X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z},
                                    {"1": 1.0 + 0.0j})
        self.assertIs(rewrite_over_independent_operators(hamiltonian), hamiltonian)

    def test_a_generic_controlled_gate_is_returned_unchanged(self):
        """``X`` is independent of the identity, so the condition already holds."""
        mps, site_ids = chain_reference(4)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1, 1: 1}, 3, PAULI_X)
        padded = hamiltonian.pad_with_identities(mps)
        self.assertIs(rewrite_over_independent_operators(padded), padded)


class TestControlledPhaseReachesTheRank(unittest.TestCase):
    """A phase target is the dependent case, and the one a circuit meets."""

    def assert_reaches_the_rank(self, num_sites, controls, target, target_matrix):
        """Builds the TTNO both ways and compares the bonds to the rank."""
        mps, site_ids = chain_reference(num_sites)
        hamiltonian = controlled_gate_hamiltonian(site_ids, controls, target,
                                                  target_matrix)
        without = TTNO.from_hamiltonian(hamiltonian, mps, method=TTNOFinder.SGE,
                                        combine_dependent_operators=False)
        with_pass = TTNO.from_hamiltonian(hamiltonian, mps, method=TTNOFinder.SGE)
        # the pass must not move the operator, so the two TTNOs are the same operator
        reference = dense_matrix(without, site_ids)
        np.testing.assert_allclose(dense_matrix(with_pass, site_ids), reference,
                                   atol=1e-12)
        # and the bond must be the operator-Schmidt rank of that operator on every cut
        ranks = operator_schmidt_ranks(reference, num_sites)
        self.assertEqual(chain_bonds(with_pass, site_ids), ranks)
        self.assertTrue(all(a <= b for a, b in
                            zip(chain_bonds(with_pass, site_ids),
                                chain_bonds(without, site_ids))))

    def test_a_controlled_phase_drops_the_bond_outside_the_controls(self):
        """``U = lambda I`` leaves a phase gate on the controls alone."""
        self.assert_reaches_the_rank(4, {0: 1, 1: 1}, 3,
                                     np.exp(0.7j) * IDENTITY)

    def test_a_controlled_phase_with_a_gap_drops_the_outer_bond(self):
        self.assert_reaches_the_rank(5, {0: 1, 3: 1}, 4,
                                     np.exp(0.7j) * IDENTITY)

    def test_a_controlled_identity_drops_to_bond_one(self):
        """``U = I`` makes the whole gate the identity."""
        self.assert_reaches_the_rank(4, {0: 1, 1: 1}, 3, IDENTITY)

    def test_a_generic_target_still_reaches_the_rank(self):
        """The branch that needs no rewrite must still come out at the rank."""
        self.assert_reaches_the_rank(4, {0: 1, 1: 1}, 3, PAULI_X)


class TestDependentOperatorsAreCombined(unittest.TestCase):
    """The projectors of a node sum to the identity, which the symbols hide."""

    def test_the_projectors_are_recognised_as_the_identity(self):
        """``I X - P0 X - P1 X + P0 Z`` is ``P0 Z``, so its bond is one."""
        mps, site_ids = chain_reference(2)
        hamiltonian = Hamiltonian()
        for frac, left, right in ((Fraction(1), "I2", "X"),
                                  (Fraction(-1), "Proj0", "X"),
                                  (Fraction(-1), "Proj1", "X"),
                                  (Fraction(1), "Proj0", "Z")):
            term = TensorProduct({site_ids[0]: left, site_ids[1]: right})
            hamiltonian.add_term((frac, "1", term))
        hamiltonian.update_mappings({"I2": IDENTITY, "X": PAULI_X, "Z": PAULI_Z,
                                     "Proj0": PROJ_0, "Proj1": PROJ_1},
                                    {"1": 1.0 + 0.0j})
        plain = TTNO.from_hamiltonian(hamiltonian, mps, method=TTNOFinder.SGE,
                                      combine_dependent_operators=False)
        ttno = TTNO.from_hamiltonian(hamiltonian, mps, method=TTNOFinder.SGE)
        self.assertEqual(chain_bonds(ttno, site_ids), [1])
        np.testing.assert_allclose(dense_matrix(ttno, site_ids),
                                   dense_matrix(plain, site_ids), atol=1e-12)

    def test_the_rewrite_keeps_the_operator(self):
        """The rewrite is exact, so the operator must not move at all."""
        mps, site_ids = chain_reference(3)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1}, 2,
                                                  np.exp(0.3j) * IDENTITY)
        padded = hamiltonian.pad_with_identities(mps)
        rewritten = rewrite_over_independent_operators(padded)
        np.testing.assert_allclose(
            rewritten.to_matrix(mps, use_padding=False).operator,
            padded.to_matrix(mps, use_padding=False).operator, atol=1e-14)
        self.assertLess(len(rewritten.terms), len(padded.terms))


class TestEveryMethodBenefits(unittest.TestCase):
    """The rewrite is a property of the Hamiltonian, so every finder gains from it."""

    def test_all_finders_reach_the_rank_on_a_controlled_phase(self):
        mps, site_ids = chain_reference(4)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1, 1: 1}, 3,
                                                  np.exp(0.7j) * IDENTITY)
        plain = TTNO.from_hamiltonian(hamiltonian, mps, method=TTNOFinder.SGE,
                                      combine_dependent_operators=False)
        ranks = operator_schmidt_ranks(dense_matrix(plain, site_ids), 4)
        for method in (TTNOFinder.SGE, TTNOFinder.SGE_PURE, TTNOFinder.TREE,
                       TTNOFinder.BIPARTITE):
            ttno = TTNO.from_hamiltonian(hamiltonian, mps, method=method)
            self.assertEqual(chain_bonds(ttno, site_ids), ranks, msg=str(method))


if __name__ == "__main__":
    unittest.main()


class TestDegenerateInput(unittest.TestCase):
    """The pass must hand back something usable however the terms cancel."""

    def test_terms_that_cancel_leave_the_hamiltonian_alone(self):
        """Nothing is left to rewrite over, so the caller keeps what it had."""
        mps, site_ids = chain_reference(2)
        hamiltonian = Hamiltonian()
        for frac, left in ((Fraction(1), "Proj0"), (Fraction(-1), "Proj0")):
            term = TensorProduct({site_ids[0]: left, site_ids[1]: "I2"})
            hamiltonian.add_term((frac, "1", term))
        # a second node operator, so the node is a candidate for the rewrite at all
        hamiltonian.add_term((Fraction(1), "1",
                              TensorProduct({site_ids[0]: "I2", site_ids[1]: "I2"})))
        hamiltonian.add_term((Fraction(-1), "1",
                              TensorProduct({site_ids[0]: "I2", site_ids[1]: "I2"})))
        hamiltonian.update_mappings({"I2": IDENTITY, "Proj0": PROJ_0, "Proj1": PROJ_1},
                                    {"1": 1.0 + 0.0j})
        rewritten = rewrite_over_independent_operators(hamiltonian)
        self.assertFalse(rewritten.is_empty())

    def test_a_single_term_is_left_alone(self):
        """One product has nothing to eliminate, so the pass is a no-op."""
        mps, site_ids = chain_reference(3)
        hamiltonian = Hamiltonian()
        hamiltonian.add_term((Fraction(1), "1",
                              TensorProduct({site_ids[0]: "X", site_ids[2]: "Z"})))
        hamiltonian.update_mappings({"X": PAULI_X, "Z": PAULI_Z}, {"1": 1.0 + 0.0j})
        self.assertIs(rewrite_over_independent_operators(hamiltonian), hamiltonian)


class TestShippedGatesAreUntouched(unittest.TestCase):
    """No gate of the library carries a dependence, so none of their TTNOs may move."""

    def test_the_multi_controlled_gate_is_returned_unchanged(self):
        """Its operation is restricted to X, Y, Z and H, none a multiple of the identity.

        This is the regression guard for the pass: the gates the library builds today are
        already independent on every node, so enabling it must leave them exactly as they
        were, bond for bond.
        """
        from pytreenet.operators.qcircuits.qgate import MultiControlledGate, QGate

        mps, site_ids = chain_reference(4)
        for operation in (QGate.PAULI_X, QGate.PAULI_Y, QGate.PAULI_Z, QGate.HADAMARD):
            gate = MultiControlledGate([site_ids[0]], [site_ids[1]], site_ids[3],
                                       operation, "MCG")
            hamiltonian = gate.as_sum_of_products()
            padded = hamiltonian.pad_with_identities(mps)
            self.assertIs(rewrite_over_independent_operators(padded), padded,
                          msg=str(operation))
            with_pass = TTNO.from_hamiltonian(hamiltonian, mps)
            without = TTNO.from_hamiltonian(hamiltonian, mps,
                                            combine_dependent_operators=False)
            self.assertEqual(chain_bonds(with_pass, site_ids),
                             chain_bonds(without, site_ids), msg=str(operation))
            np.testing.assert_allclose(dense_matrix(with_pass, site_ids),
                                       dense_matrix(without, site_ids), atol=1e-12)


class TestNearlyDependentOperators(unittest.TestCase):
    """A dependence that only holds over the rationals must be refused."""

    def test_a_nearly_dependent_node_is_left_alone(self):
        """A dependence that holds only over the rationals leaves the node unrewritten.

        The first four operators span a four-dimensional space only through rounding, so
        the fifth is dependent on them with exact coordinates up to ``5e16``, and rebuilding
        it in double precision misses it by 11 times its largest entry. The node is left
        alone, so the elimination keeps the bond it would have had.
        """
        rng = np.random.default_rng(11)

        def diagonal():
            return np.diag(rng.normal(size=4) + 1j * rng.normal(size=4))

        base = diagonal()
        drift = diagonal()
        nearly = [base + k * 1e-9 * drift for k in range(4)]
        nearly.append(diagonal())
        flat = np.array([op.ravel() for op in nearly])
        self.assertLess(np.linalg.matrix_rank(flat, tol=1e-10), len(nearly))
        self.assertLess(np.linalg.matrix_rank(flat[:4], tol=1e-10), 4)

        mps, site_ids = chain_reference(2)
        hamiltonian = Hamiltonian()
        names = [f"near{k}" for k in range(len(nearly))]
        for k, name in enumerate(names):
            term = TensorProduct({site_ids[0]: name, site_ids[1]: "I2"})
            hamiltonian.add_term((Fraction(k + 1), "1", term))
        hamiltonian.update_mappings(
            {"I2": np.eye(4, dtype=complex),
             **{name: op.astype(complex) for name, op in zip(names, nearly)}},
            {"1": 1.0 + 0.0j})
        padded = hamiltonian.pad_with_identities(mps)
        # the node is a candidate, five distinct operators of which only two are a basis
        self.assertIs(rewrite_over_independent_operators(padded), padded)

    def test_the_lindbladian_ttno_equals_its_dense_generator(self):
        """The fused Pauli Lindbladian TTNO is the dense generator on 3 and 4 qubits.

        Its float operators are nearly dependent on a node, which the guard must refuse
        for the operator to survive the rewrite.
        """
        for num_qubits in (3, 4):
            with self.subTest(n=num_qubits):
                rng = np.random.default_rng(99)
                model = lindbladian_model(rng, num_qubits)
                state = pauli_product_mps([random_ket(rng) for _ in range(num_qubits)])
                ids = pauli_site_ids(state)
                ttno = pauli_lindbladian_ttno(state, *model)
                dense = pauli_dense_lindbladian(num_qubits, *model)
                np.testing.assert_allclose(ttno_dense(ttno, ids), dense, atol=1e-10)


class TestTheBetterDiagramIsKept(unittest.TestCase):
    """The rewrite makes the bound tight, the elimination still has to reach it."""

    def symbolic_lindbladian(self, num_qubits):
        """Returns the Lindbladian of a transverse field chain and its site order."""
        qubits = [f"q{i}" for i in range(num_qubits)]
        system = Hamiltonian()
        for i in range(num_qubits - 1):
            system.add_term((Fraction(1), "J",
                             TensorProduct({qubits[i]: "Z", qubits[i + 1]: "Z"})))
        for qubit in qubits:
            for symbol in ("X", "Y"):
                system.add_term((Fraction(1), "h", TensorProduct({qubit: symbol})))
        system.update_mappings({"X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z,
                                "I2": IDENTITY},
                               {"J": 1.0 + 0.0j, "h": 0.5 + 0.0j})
        jumps = [(Fraction(1), "g", TensorProduct({qubit: "Lm"})) for qubit in qubits]
        lindbladian = generate_lindbladian(system, jumps, {"Lm": SIGMA_MINUS},
                                           {"g": 0.1 + 0.0j})
        site_ids = [f"{qubit}{suffix}" for qubit in qubits
                    for suffix in ("_ket", "_bra")]
        return lindbladian, site_ids

    def test_a_rewrite_that_costs_a_bond_is_discarded(self):
        """On these terms the elimination does worse once they are rewritten.

        The rewritten terms carry the tight bound, and the elimination reaches 6 on the
        middle cut where the operator-Schmidt rank is 5, while the terms as given reach
        5. The diagram with the smaller total bond must be the one that is kept.
        """
        lindbladian, site_ids = self.symbolic_lindbladian(3)
        tensors = [np.ones((1, 1, 2), dtype=complex) for _ in site_ids]
        tensors[0] = np.ones((1, 2), dtype=complex)
        tensors[-1] = np.ones((1, 2), dtype=complex)
        mps = MatrixProductState.from_tensor_list(tensors,
                                                  node_prefix=list(site_ids),
                                                  root_site=0)
        padded = lindbladian.pad_with_identities(mps)
        rewritten = rewrite_over_independent_operators(padded)
        self.assertIsNot(rewritten, padded)  # the terms really are rewritten
        plain = StateDiagram.from_hamiltonian(padded, mps, TTNOFinder.SGE)
        other = StateDiagram.from_hamiltonian(rewritten, mps, TTNOFinder.SGE)
        self.assertGreater(sum(other.bond_dimensions().values()),
                           sum(plain.bond_dimensions().values()))
        ttno = TTNO.from_hamiltonian(lindbladian, mps, method=TTNOFinder.SGE)
        without = TTNO.from_hamiltonian(lindbladian, mps, method=TTNOFinder.SGE,
                                        combine_dependent_operators=False)
        self.assertEqual(chain_bonds(ttno, site_ids),
                         chain_bonds(without, site_ids))
        reference = dense_matrix(without, site_ids)
        self.assertEqual(chain_bonds(ttno, site_ids),
                         operator_schmidt_ranks(reference, len(site_ids)))
        np.testing.assert_allclose(dense_matrix(ttno, site_ids), reference, atol=1e-12)

    def test_a_rewrite_that_saves_a_bond_is_kept(self):
        """The guard must not throw away the reduction it is there to protect."""
        mps, site_ids = chain_reference(4)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1, 1: 1}, 3,
                                                 np.exp(0.7j) * IDENTITY)
        ttno = TTNO.from_hamiltonian(hamiltonian, mps, method=TTNOFinder.SGE)
        self.assertEqual(chain_bonds(ttno, site_ids), [2, 1, 1])
