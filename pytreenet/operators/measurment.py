"""
This module implements the measurement class
"""
from __future__ import annotations
from typing import Self, Iterable
from itertools import product

import numpy as np

from ..ttns.ttns import TTNS
from .common_operators import projector
from .tensorproduct import TensorProduct

class Outcome:
    """
    A class to represent a measurement outcome.
    """

    def __init__(self,
                 node_ids: Iterable[str],
                 state_vals: Iterable[int]
                 ) -> None:
        """
        Initializes an Outcome instance.

        Args:
            node_ids (Iterable[str]): An iterable of node IDs corresponding to the measurement outcome.
            state_vals (Iterable[int]): An iterable of state values corresponding to the measurement outcome.
        """
        self.data = frozenset(zip(node_ids, state_vals))

    @classmethod
    def from_iters(cls,
                   node_ids: Iterable[str],
                   state_vals: Iterable[int]
                   ) -> Self:
        """
        Creates an Outcome from iterables of node IDs and state values.

        Args:
            node_ids (Iterable[str]): An iterable of node IDs corresponding to the measurement outcome.
            state_vals (Iterable[int]): An iterable of state values corresponding to the measurement outcome.

        Returns:
            Self: The created Outcome instance.
        """
        return cls(node_ids, state_vals)
    
    def otimes(self, other: Outcome) -> Self:
        """
        Kronecker product of two Outcomes.
        """
        this_node_ids = set(node_id for node_id, _ in self.data)
        other_node_ids = set(node_id for node_id, _ in other.data)
        overlapping_keys = this_node_ids.intersection(other_node_ids)
        if overlapping_keys:
            errstr = f"Cannot combine Outcomes with overlapping node IDs: {overlapping_keys}!"
            raise ValueError(errstr)
        new_data = self.data.union(other.data)
        new = self.__class__([], [])
        new.data = frozenset(new_data)
        return new

    def as_dict(self):
        return dict(self.data)
    
    @classmethod
    def empty(cls) -> Self:
        """
        Creates an empty Outcome.

        Returns:
            Self: An empty Outcome instance.
        """
        return cls.from_iters([], [])
    
    def __hash__(self) -> int:
        return hash(self.data)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Outcome):
            return False
        return self.data == other.data

class Measurement:
    """
    A class to represent a measurement on a Tree Tensor Network.
    """

    def __init__(self,
                 node_ids: list[str],
                 renormalize: bool = True,
                 num_thresholds: float = 1e-10,
                 seed: int | None = None
                 ) -> None:
        """
        Initializes a Measurement instance.

        Args:
            node_ids (list[str], optional): A list of node IDs to be measured.
            renormalize (bool, optional): Whether to renormalize the state after
                applying the measurement. Defaults to True.
            num_thresholds (float, optional): The numerical thresholds to
                consider things accurate as. Defaults to 1e-10.
            seed (Union[int, None], optional): A seed for the random number generator
                used in the measurement process. Defaults to None.
        """
        self.node_ids = node_ids
        self.renormalize = renormalize
        self.num_thresholds = num_thresholds
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_list(cls,
                  node_ids: list[str],
                  **kwargs
                  ) -> Self:
        """
        Creates a Measurement from a list of node IDs.

        Args:
            node_ids (list[str]): A list of node IDs to be measured.
            **kwargs: Additional keyword arguments for the Measurement constructor.

        Returns:
            Self: The created Measurement instance.
        """
        return cls(node_ids, **kwargs)

    @classmethod
    def empty(cls,
              **kwargs) -> Self:
        """
        Creates an empty Measurement.

        Returns:
            Self: An empty Measurement instance.
        """
        return cls(**kwargs)

    def is_empty(self) -> bool:
        """
        Checks if the Measurement is empty.

        Returns:
            bool: True if the Measurement is empty, False otherwise.
        """
        return len(self.node_ids) == 0

    def otimes(self,
               other: Measurement
               ) -> Self:
        """
        Combines two Measurements using the outer tensor product.

        Args:
            other (Measurement): The other Measurement to combine with.
        
        Returns:
            Measurement: A new Measurement representing the combined measurements.
        
        Raises:
            ValueError: If there are overlapping node IDs in the two Measurements or if
                        the two measurements don't agree on renormalization.
        """
        overlapping_keys = set(self.node_ids).intersection(set(other.node_ids))
        if overlapping_keys:
            errstr = f"Cannot combine Measurements with overlapping node IDs: {overlapping_keys}!"
            raise ValueError(errstr)
        if self.renormalize != other.renormalize:
            errstr = "Cannot combine Measurements with different renormalization settings!"
            raise ValueError(errstr)
        new = self.__class__(self.node_ids + other.node_ids, renormalize=self.renormalize)
        return new

    def system_size(self) -> int:
        """
        Returns the number of nodes involved in the Measurement.

        Returns:
            int: The number of nodes.
        """
        return len(self.node_ids)

    def __eq__(self,
               other: object
               ) -> bool:
        if not isinstance(other, Measurement):
            return False
        return self.renormalize == other.renormalize and self.node_ids == other.node_ids

    def apply(self,
              state: TTNS
              ) -> tuple[Outcome, float]:
        """
        Apply the measurement to the given state.

        Args:
            state (TTNS): The state to which the measurement should be applied.

        Returns:
            tuple[Outcome, float]: A tuple containing the measurement outcome
                and the probability of that outcome.
        """
        dims = {node_id: state.nodes[node_id].open_dimension()
                for node_id in self.node_ids}
        projections = {node_id: [projector(dims[node_id],outcome)
                                 for outcome in range(dims[node_id])]
                        for node_id in self.node_ids}
        # We check all possible outcomes until the sum of the outcome
        # probabilities exceeds a random threshold. The last outcome is the
        # one we will use for the measurement.
        thresh = self.rng.random()
        prob_passed = 0.0
        iterator = product(*[range(dims[node_id]) for node_id in self.node_ids])
        for outcome in iterator:
            tp = TensorProduct()
            for node_id, outcome_i in zip(self.node_ids, outcome):
                tp.add_operator(node_id, projections[node_id][outcome_i])
            prob = state.tensor_product_expectation_value(tp)
            # This is a complex number, so we need to check its validity.
            if prob.imag > self.num_thresholds:
                errstr = f"Measurement probability has a significant imaginary part: {prob}!"
                raise ValueError(errstr)
            prob = prob.real
            prob_passed += prob
            if prob_passed > thresh:
                # This is the outcome to be used!
                state.apply_operator(tp)
                if self.renormalize:
                    state.normalize(np.sqrt(prob))
                if state.orthogonality_center_id:
                    state.canonical_form(state.orthogonality_center_id)
                return Outcome(self.node_ids, outcome), prob
        # If we get here, something went wrong with the probabilities. We can
        # raise an error.
        errstr = f"Measurement probabilities do not sum up to 1! Total probability: {prob_passed}."
        raise ValueError(errstr)

class _SingleMCUnitary(Measurement):
    """
    A class to represent a measurement-controlled unitary operation on a Tree Tensor Network.
    """
    
    def __init__(self,
                 node_ids: list[str],
                 unitaries: dict[Outcome, TensorProduct],
                 non_ex_is_identity: bool = True,
                 **kwargs
                 ) -> None:
        """
        Initializes a MeasurementControlledUnitary instance.

        Args:
            node_ids (list[str]): A list of node IDs to be measured.
            unitaries (dict[Outcome, TensorProduct]): A dictionary mapping
                measurement outcomes to unitary operations.
            non_ex_is_identity (bool, optional): Whether to treat outcomes not
                explicitly in the unitaries dictionary as identity operations.
                Defaults to True.
            **kwargs: Additional keyword arguments for the Measurement
                constructor.
        """
        super().__init__(node_ids, **kwargs)
        self.unitaries = unitaries
        self.non_ex_is_identity = non_ex_is_identity

    def apply(self,
              state: TTNS
              ) -> tuple[Outcome, float]:
        """
        Apply the measurement-controlled unitary operation to the given state.

        Args:
            state (TTNS): The state to which the operation should be applied.

        Returns:
            tuple[Outcome, float]: A tuple containing the measurement outcome
                and the probability of that outcome.
        """
        outcome, prob = super().apply(state)
        unitary = self.unitaries.get(outcome)
        if unitary is not None:
            state.apply_operator(unitary)
        elif unitary is None and not self.non_ex_is_identity:
            raise ValueError(f"No unitary found for outcome: {outcome}!")
        return outcome, prob

class MeasurementControlledUnitary(Measurement):
    """
    A class to represent a measurement-controlled unitary operation on a Tree Tensor Network.
    """
    
    def __init__(self,
                 node_ids: list[str],
                 unitaries: dict[Outcome, TensorProduct],
                 non_ex_is_identity: bool = True,
                 **kwargs
                 ) -> None:
        """
        Initializes a MeasurementControlledUnitary instance.

        Args:
            node_ids (list[str]): A list of node IDs to be measured.
            unitaries (dict[Outcome, TensorProduct]): A dictionary mapping
                measurement outcomes to unitary operations.
            non_ex_is_identity (bool, optional): Whether to treat outcomes not
                explicitly in the unitaries dictionary as identity operations.
                Defaults to True.
            **kwargs: Additional keyword arguments for the Measurement
                constructor.
        """
        super().__init__(node_ids, **kwargs)
        # Doing it this way allows us to easier do otimes.
        # If we stored all the controlled unitaries in a single dictionary
        # we would have to otimes all tensor product for every combination
        # of outcomes, which becomes cubersome.
        # Since the measurement parts of the circuit that are otimesed
        # together are independent, we can just store the unitaries for each
        # measurement part separately and then combine them when we apply the
        # operation.
        self.parts = [_SingleMCUnitary(node_ids,
                                        unitaries,
                                        non_ex_is_identity,
                                        **kwargs
                                        )]

    def otimes(self, other: MeasurementControlledUnitary) -> Self:
        """
        Kronecker product of two MeasurementControlledUnitary operations.
        """
        overlapping_keys = set(self.node_ids).intersection(set(other.node_ids))
        if overlapping_keys:
            errstr = f"Cannot combine MeasurementControlledUnitary with overlapping node IDs: {overlapping_keys}!"
            raise ValueError(errstr)
        if self.renormalize != other.renormalize:
            errstr = "Cannot combine MeasurementControlledUnitary with different renormalization settings!"
            raise ValueError(errstr)
        new = self.__class__(self.node_ids + other.node_ids,
                             {},
                             renormalize=self.renormalize)
        new.parts = self.parts + other.parts
        return new

    def apply(self,
              state: TTNS
              ) -> tuple[Outcome, float]:
        """
        Apply the measurement-controlled unitary operation to the given state.

        Args:
            state (TTNS): The state to which the operation should be applied.

        Returns:
            tuple[Outcome, float]: A tuple containing the measurement outcome
                and the probability of that outcome.
        """
        outcomes_probs = [part.apply(state) for part in self.parts]
        outcome = Outcome.empty()
        prob = 1.0
        for outcome_i, prob_i in outcomes_probs:
            outcome = outcome.otimes(outcome_i)
            prob *= prob_i
        return outcome, prob
