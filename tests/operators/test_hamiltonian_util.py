"""
Tests for rewriting a Hamiltonian over operators independent on every node.

The tests pin that the rewrite leaves the operator unchanged, that it leaves a Hamiltonian
alone when there is nothing to rewrite or when a dependence does not hold in double
precision, that the bond of the TTNO built from the rewritten terms is the operator-Schmidt
rank, and that each rewrite policy builds from the presentation it promises.
"""
import unittest
from fractions import Fraction

import numpy as np

from pytreenet.operators.common_operators import (pauli_matrices,
                                                  projector,
                                                  spin_jumps)
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.hamiltonian_util import rewrite_over_independent_operators
from pytreenet.operators.lindbladian import generate_lindbladian
from pytreenet.operators.qcircuits.qgate import MultiControlledGate, QGate
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.random import random_nearly_dependent_diagonals
from pytreenet.random.random_special_ttns import random_mps
from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.ttno.state_diagram import OperatorRewrite, StateDiagram, TTNOFinder
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator as TTNO

PAULI_X, PAULI_Y, PAULI_Z = pauli_matrices()
PROJ_0, PROJ_1 = projector(2, 0), projector(2, 1)
# the library calls [[0, 1], [0, 0]] the raising operator
RAISING, LOWERING = spin_jumps()


def chain_reference(num_sites: int, phys_dim: int = 2):
    """Returns an MPS to use as the reference tree, and its site identifiers in order."""
    mps = random_mps(num_sites, phys_dim, 2, root_site=0)
    site_ids = [f"site{i}" for i in range(num_sites)]
    assert set(site_ids) == set(mps.nodes), "the MPS does not use the expected site ids"
    return mps, site_ids


def chain_with_ids(site_ids, phys_dim: int = 2):
    """Returns a chain to use as the reference tree, carrying the given identifiers."""
    tensors = [np.ones((1, 1, phys_dim), dtype=complex) for _ in site_ids]
    tensors[0] = np.ones((1, phys_dim), dtype=complex)
    tensors[-1] = np.ones((1, phys_dim), dtype=complex)
    return MatrixProductState.from_tensor_list(tensors,
                                               node_prefix=list(site_ids),
                                               root_site=0)


def controlled_gate_hamiltonian(site_ids, controls, target, target_matrix):
    """
    The three strings of a controlled gate, as MultiControlledGate builds them.

    The library restricts its own gate to X, Y, Z and H, so a target carrying a multiple
    of the identity has to be built here.

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
    hamiltonian.update_mappings({"Proj0": PROJ_0, "Proj1": PROJ_1, "U": target_matrix},
                                {"1": 1.0 + 0.0j})
    hamiltonian.include_identities(2)
    return hamiltonian


def symbolic_lindbladian(num_qubits: int):
    """Returns the Lindbladian of a transverse field chain and its site order."""
    qubits = [f"q{i}" for i in range(num_qubits)]
    system = Hamiltonian()
    for i in range(num_qubits - 1):
        system.add_term((Fraction(1), "J",
                         TensorProduct({qubits[i]: "Z", qubits[i + 1]: "Z"})))
    for qubit in qubits:
        for symbol in ("X", "Y"):
            system.add_term((Fraction(1), "h", TensorProduct({qubit: symbol})))
    system.update_mappings({"X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z},
                           {"J": 1.0 + 0.0j, "h": 0.5 + 0.0j})
    system.include_identities(2)
    jumps = [(Fraction(1), "g", TensorProduct({qubit: "Lm"})) for qubit in qubits]
    lindbladian = generate_lindbladian(system, jumps, {"Lm": LOWERING},
                                       {"g": 0.1 + 0.0j})
    site_ids = [f"{qubit}{suffix}" for qubit in qubits
                for suffix in ("_ket", "_bra")]
    return lindbladian, site_ids


def cancelling_hamiltonian(site_ids):
    """
    Returns terms that cancel once the identity is rewritten over the projectors.

    ``I (x) X - P0 (x) X - P1 (x) X`` is the zero operator, which only shows after the
    first node is rewritten, since the three symbols there are distinct.
    """
    hamiltonian = Hamiltonian()
    for frac, left in ((Fraction(1), "I2"), (Fraction(-1), "Proj0"),
                       (Fraction(-1), "Proj1")):
        hamiltonian.add_term((Fraction(frac), "1",
                              TensorProduct({site_ids[0]: left, site_ids[1]: "X"})))
    hamiltonian.update_mappings({"Proj0": PROJ_0, "Proj1": PROJ_1, "X": PAULI_X},
                                {"1": 1.0 + 0.0j})
    hamiltonian.include_identities(2)
    return hamiltonian


def dense_matrix(ttno, site_ids):
    """Returns the dense operator of a TTNO with ``site_ids[0]`` the most significant leg."""
    matrix, order = ttno.as_matrix(order=list(site_ids))
    assert order == list(site_ids), f"as_matrix returned the order {order}"
    return matrix


def schmidt_rank_bonds(matrix, site_ids, phys_dim: int = 2, tol: float = 1e-10):
    """
    Returns the operator-Schmidt rank of a dense operator, keyed by the edge it bounds.

    The rank is the smallest bond dimension any representation of the operator can carry
    on that edge, so it is what the bond dimensions of an optimal TTNO are compared to.
    """
    num_sites = len(site_ids)
    tensor = np.asarray(matrix).reshape([phys_dim] * (2 * num_sites))
    bonds = {}
    for cut in range(1, num_sites):
        permutation = [*range(cut),
                       *range(num_sites, num_sites + cut),
                       *range(cut, num_sites),
                       *range(num_sites + cut, 2 * num_sites)]
        reshuffled = tensor.transpose(permutation).reshape(phys_dim ** (2 * cut), -1)
        singular = np.linalg.svd(reshuffled, compute_uv=False)
        bonds[(site_ids[cut - 1], site_ids[cut])] = int((singular > tol).sum())
    return bonds


def diagram_cost(hamiltonian, reference, rewrite, method=TTNOFinder.SGE):
    """Returns the bond cost of the diagram one policy builds."""
    diagram = StateDiagram.from_hamiltonian(hamiltonian, reference, method,
                                            rewrite=rewrite)
    return diagram.bond_cost()


class TestRewriteLeavesIndependentOperatorsAlone(unittest.TestCase):
    """There must be no rewrite where there is no dependence to remove."""

    def test_an_independent_hamiltonian_is_returned_unchanged(self):
        """Pauli matrices are independent, and one term alone has nothing to combine."""
        _, site_ids = chain_reference(4)
        pauli_sum = Hamiltonian()
        for operator in ("X", "Y", "Z"):
            term = TensorProduct({site_ids[0]: operator, site_ids[2]: operator})
            pauli_sum.add_term((Fraction(1), "1", term))
        pauli_sum.update_mappings({"X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z},
                                  {"1": 1.0 + 0.0j})
        self.assertIs(rewrite_over_independent_operators(pauli_sum), pauli_sum)

        single = Hamiltonian()
        single.add_term((Fraction(1), "1",
                         TensorProduct({site_ids[0]: "X", site_ids[2]: "Z"})))
        single.update_mappings({"X": PAULI_X, "Z": PAULI_Z}, {"1": 1.0 + 0.0j})
        self.assertIs(rewrite_over_independent_operators(single), single)

    def test_the_shipped_controlled_gates_are_returned_unchanged(self):
        """The library restricts its gate to operations independent of the identity.

        This is the regression guard. No gate the library builds today carries a
        dependence, so none of their TTNOs may move under any policy.
        """
        mps, site_ids = chain_reference(4)
        for operation in (QGate.PAULI_X, QGate.PAULI_Y, QGate.PAULI_Z, QGate.HADAMARD):
            gate = MultiControlledGate([site_ids[0]], [site_ids[1]], site_ids[3],
                                       operation, "MCG")
            padded = gate.as_sum_of_products().pad_with_identities(mps)
            self.assertIs(padded.rewrite_over_independent_operators(), padded,
                          msg=str(operation))

    def test_a_dependence_that_only_holds_over_the_rationals_is_refused(self):
        """An exact dependence between float operators can be an artefact of their digits.

        At a condition number of 1e16 the kept operators are no basis in double
        precision, so the node is left alone whatever the tolerance. At 1e6 they are,
        and the tolerance decides.
        """
        operators = random_nearly_dependent_diagonals(4, 1e16, seed=11)
        hopeless = self.node_hamiltonian(operators)
        for tol in (1e-13, 1e-8, 1e-2):
            self.assertIs(hopeless.rewrite_over_independent_operators(tol), hopeless,
                          msg=f"tol {tol}")

        operators = random_nearly_dependent_diagonals(4, 1e6, seed=11)
        decidable = self.node_hamiltonian(operators)
        self.assertIs(decidable.rewrite_over_independent_operators(1e-13), decidable)
        self.assertIsNot(decidable.rewrite_over_independent_operators(1e-8), decidable)

    def node_hamiltonian(self, operators):
        """Returns a Hamiltonian carrying one operator of a set on its first node."""
        _, site_ids = chain_reference(2, phys_dim=len(operators[0]))
        hamiltonian = Hamiltonian()
        names = [f"op{index}" for index in range(len(operators))]
        for index, name in enumerate(names):
            hamiltonian.add_term((Fraction(index + 1), "1",
                                  TensorProduct({site_ids[0]: name,
                                                 site_ids[1]: "I4"})))
        hamiltonian.update_mappings(
            {name: operator.astype(complex) for name, operator in zip(names, operators)},
            {"1": 1.0 + 0.0j})
        hamiltonian.include_identities(len(operators[0]))
        return hamiltonian


class TestRewriteReachesTheOperatorSchmidtRank(unittest.TestCase):
    """Once the operators are independent, the bond must be the true minimum."""

    def assert_reaches_the_rank(self, hamiltonian, reference, site_ids,
                               method=TTNOFinder.SGE):
        """The rewritten TTNO carries the same operator at its operator-Schmidt rank."""
        given = TTNO.from_hamiltonian(hamiltonian, reference, method=method)
        rewritten = TTNO.from_hamiltonian(hamiltonian, reference, method=method,
                                          rewrite=OperatorRewrite.ALWAYS)
        expected = dense_matrix(given, site_ids)
        np.testing.assert_allclose(dense_matrix(rewritten, site_ids), expected,
                                   atol=1e-12)
        self.assertEqual(rewritten.bond_dims(),
                         schmidt_rank_bonds(expected, site_ids), msg=str(method))

    def test_a_controlled_identity_reaches_the_rank(self):
        """The gate collapses to the identity, the extreme case of a dependence."""
        mps, site_ids = chain_reference(4)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1, 1: 1}, 3,
                                                  np.eye(2, dtype=complex))
        self.assert_reaches_the_rank(hamiltonian, mps, site_ids)

    def test_dependent_projectors_reach_the_rank(self):
        """The dependence is between the identity and the projectors summing to it."""
        mps, site_ids = chain_reference(2)
        hamiltonian = Hamiltonian()
        for frac, left, right in ((Fraction(1), "I2", "X"),
                                  (Fraction(-1), "Proj0", "X"),
                                  (Fraction(-1), "Proj1", "X"),
                                  (Fraction(1), "Proj0", "Z")):
            hamiltonian.add_term((frac, "1", TensorProduct({site_ids[0]: left,
                                                            site_ids[1]: right})))
        hamiltonian.update_mappings({"Proj0": PROJ_0, "Proj1": PROJ_1,
                                    "X": PAULI_X, "Z": PAULI_Z},
                                   {"1": 1.0 + 0.0j})
        hamiltonian.include_identities(2)
        self.assert_reaches_the_rank(hamiltonian, mps, site_ids)

    def test_every_finder_reaches_the_rank(self):
        """The condition is a property of the Hamiltonian, not of the finder.

        The tree comparison method is left out because it does not read the
        prefactor of a term, so it does not represent a Hamiltonian whose terms carry
        signs, which every controlled gate does.
        """
        mps, site_ids = chain_reference(4)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1, 1: 1}, 3,
                                                  np.exp(0.7j) * np.eye(2, dtype=complex))
        for method in (TTNOFinder.SGE, TTNOFinder.SGE_PURE, TTNOFinder.BIPARTITE):
            self.assert_reaches_the_rank(hamiltonian, mps, site_ids, method=method)


class TestRewriteKeepsTheOperator(unittest.TestCase):
    """The rewritten Hamiltonian must be another presentation of the same operator."""

    def test_the_rewritten_hamiltonian_is_the_same_operator(self):
        """The arithmetic is exact, so the two agree to machine precision.

        The rewrite also names no operator the Hamiltonian did not carry, so the
        conversion dictionary is unchanged and only the coefficient mapping grows.
        """
        mps, site_ids = chain_reference(4)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1, 1: 1}, 3,
                                                  np.exp(0.7j) * np.eye(2, dtype=complex))
        padded = hamiltonian.pad_with_identities(mps)
        rewritten = padded.rewrite_over_independent_operators()
        self.assertIsNot(rewritten, padded)
        np.testing.assert_allclose(rewritten.to_matrix(mps).operator,
                                   padded.to_matrix(mps).operator, atol=1e-12)
        self.assertEqual(set(rewritten.conversion_dictionary),
                         set(padded.conversion_dictionary))
        self.assertTrue(set(padded.coeffs_mapping)
                        <= set(rewritten.coeffs_mapping))


class TestHamiltonianMethod(unittest.TestCase):
    """The method on the class is the entry point, so it must forward everything."""

    def test_the_method_agrees_with_the_function(self):
        """Both reach the same Hamiltonian from the same terms."""
        mps, site_ids = chain_reference(4)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1, 1: 1}, 3,
                                                  np.exp(0.7j) * np.eye(2, dtype=complex))
        padded = hamiltonian.pad_with_identities(mps)
        by_method = padded.rewrite_over_independent_operators()
        by_function = rewrite_over_independent_operators(padded)
        self.assertEqual(by_method, by_function)
        self.assertTrue(by_method.compare_dicts(by_function))

    def test_the_method_forwards_the_tolerance(self):
        """The same operators are rewritten or refused depending on the tolerance."""
        operators = random_nearly_dependent_diagonals(4, 1e6, seed=11)
        _, site_ids = chain_reference(2, phys_dim=4)
        hamiltonian = Hamiltonian()
        names = [f"op{index}" for index in range(len(operators))]
        for index, name in enumerate(names):
            hamiltonian.add_term((Fraction(index + 1), "1",
                                  TensorProduct({site_ids[0]: name,
                                                 site_ids[1]: "I4"})))
        hamiltonian.update_mappings(
            {name: operator.astype(complex) for name, operator in zip(names, operators)},
            {"1": 1.0 + 0.0j})
        hamiltonian.include_identities(4)
        self.assertIs(hamiltonian.rewrite_over_independent_operators(1e-13),
                      hamiltonian)
        self.assertIsNot(hamiltonian.rewrite_over_independent_operators(1e-8),
                         hamiltonian)


class TestRewritePolicy(unittest.TestCase):
    """Each policy must build from the presentation it promises."""

    def test_the_default_policy_does_not_modify_the_hamiltonian(self):
        """A caller who asks for nothing gets a diagram of the terms they passed in."""
        mps, site_ids = chain_reference(4)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1, 1: 1}, 3,
                                                  np.exp(0.7j) * np.eye(2, dtype=complex))
        padded = hamiltonian.pad_with_identities(mps)
        self.assertIsNot(padded.rewrite_over_independent_operators(), padded)
        by_default = StateDiagram.from_hamiltonian(padded, mps, TTNOFinder.SGE)
        self.assertIs(by_default.hamiltonian, padded)
        always = StateDiagram.from_hamiltonian(padded, mps, TTNOFinder.SGE,
                                               rewrite=OperatorRewrite.ALWAYS)
        self.assertIsNot(always.hamiltonian, padded)

    def test_the_better_policy_takes_the_smaller_cost_where_the_rewrite_wins(self):
        """On a controlled phase the rewritten terms are the cheaper presentation."""
        mps, site_ids = chain_reference(4)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1, 1: 1}, 3,
                                                  np.exp(0.7j) * np.eye(2, dtype=complex))
        padded = hamiltonian.pad_with_identities(mps)
        costs = [diagram_cost(padded, mps, policy)
                 for policy in (OperatorRewrite.NONE, OperatorRewrite.ALWAYS)]
        self.assertEqual(diagram_cost(padded, mps, OperatorRewrite.BETTER), min(costs))

    def test_the_better_policy_takes_the_smaller_cost_where_the_rewrite_loses(self):
        """The construction can do worse on the rewritten terms, and then they are dropped.

        The rewrite makes the bound tight, attaining it is left to the construction, so
        the policy compares what was built rather than trusting the rewrite.
        """
        lindbladian, site_ids = symbolic_lindbladian(3)
        mps = chain_with_ids(site_ids)
        padded = lindbladian.pad_with_identities(mps)
        self.assertIsNot(padded.rewrite_over_independent_operators(), padded)
        costs = [diagram_cost(padded, mps, policy)
                 for policy in (OperatorRewrite.NONE, OperatorRewrite.ALWAYS)]
        self.assertEqual(diagram_cost(padded, mps, OperatorRewrite.BETTER), min(costs))
        # whichever presentation wins, the TTNO is the same operator
        given = TTNO.from_hamiltonian(lindbladian, mps, method=TTNOFinder.SGE)
        better = TTNO.from_hamiltonian(lindbladian, mps, method=TTNOFinder.SGE,
                                       rewrite=OperatorRewrite.BETTER)
        np.testing.assert_allclose(dense_matrix(better, site_ids),
                                   dense_matrix(given, site_ids), atol=1e-12)


class TestCancellingTerms(unittest.TestCase):
    """Terms that cancel once rewritten leave no operator to represent."""

    def test_a_rewrite_that_cancels_every_term_gives_an_empty_hamiltonian(self):
        """The rewrite reports the operator it found, which here is zero."""
        _, site_ids = chain_reference(2)
        hamiltonian = cancelling_hamiltonian(site_ids)
        rewritten = hamiltonian.rewrite_over_independent_operators()
        self.assertTrue(rewritten.is_empty())
        self.assertFalse(hamiltonian.is_empty())

    def test_a_hamiltonian_without_terms_has_no_state_diagram(self):
        """An empty Hamiltonian is rejected, however it came to be empty."""
        mps, site_ids = chain_reference(2)
        with self.assertRaises(ValueError):
            StateDiagram.from_hamiltonian(Hamiltonian(), mps, TTNOFinder.SGE)
        hamiltonian = cancelling_hamiltonian(site_ids)
        with self.assertRaises(ValueError):
            StateDiagram.from_hamiltonian(hamiltonian, mps, TTNOFinder.SGE,
                                          rewrite=OperatorRewrite.ALWAYS)
        # comparing two presentations still has the terms as given to fall back on
        better = StateDiagram.from_hamiltonian(hamiltonian, mps, TTNOFinder.SGE,
                                               rewrite=OperatorRewrite.BETTER)
        self.assertIs(better.hamiltonian, hamiltonian)


if __name__ == "__main__":
    unittest.main()
