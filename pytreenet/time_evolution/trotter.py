from __future__ import annotations
from typing import List, Union, List, Tuple

import numpy as np

from scipy.linalg import expm

from ..operators.operator import NumericOperator
from ..operators.common_operators import swap_gate

class SWAPlist(list):

    def __init__(self, swap_list):
        """
        Represent a consecutive application of SWAPs of neighbouring nodes.

        Parameters
        ----------
        tuples : list of tuples
            Each tuple contains exactly two identifiers of neighbouring nodes.
            The order of the list is the order in which SWAP gates would be
            applied. Note: neighbouring sites have to have the same physical
            dimension to be swapped.

        """
        for pair in swap_list:
            if not len(pair) == 2:
                raise ValueError("SWAPs can only happen between exactly two nodes!")

        list.__init__(self, swap_list)

    def is_compatible_with_ttn(self, ttn):
        """
        Checks, if SWAPlist is compatible with a given TreeTensorNetwork. This
        means it checks if all nodes in the SWAP-list are actually neighbours
        with the same open leg dimension.

        Parameters
        ----------
        ttn : TreeTensorNetwork
            A TTN for which to check compatability.

        Returns
        -------
        compatible: bool
        True if the SWAPlist is compatible with the TTN and False,
        if not.

        """
        for swap_pair in self:
            # Check if the first swap node is in the TTN
            if swap_pair[0] not in ttn.nodes:
                return False

            # If it is check, if the other is actually connected and thus also
            #  in the TTN
            node1 = ttn.nodes[swap_pair[0]]
            if not swap_pair[1] not in node1.neighbouring_nodes(with_legs=False):
                return False

            # Finally check if both have the same total physical dimension.
            node2 = ttn.nodes[swap_pair[1]]
            if node1.open_dimension() != node2.open_dimension():
                return False

        return True

    def into_operators(self, ttn: Union[TreeTensorNetwork, None] = None,
                       dim: Union[int, None] = None) -> List[NumericOperator]:
        """
        Turns the list of abstract swaps into a list of numeric operators.

        Args:
            ttn (TreeTensorNetwork): A tree tensor network from which the dimensions can
             be determined. Default to None.
            dim (Union[int, None], optional): Can be given, if all nodes that have
             to be swapped have the same dimension. Defaults to None.

        Returns:
            List[NumericOperator]: A list of numeric operators corresponding to the
             swaps defined in this instance.

        Raises:
            ValueError: If ttn and dim are both None.
        """
        if ttn is None and dim is None:
            errstr = "`ttn` and `dim` cannot both be `None`!"
            raise ValueError(errstr)
        operator_list = []
        if dim is not None:
            swap_matrix = swap_gate(dimension=dim)
        for swap_pair in self:
            if ttn is not None:
                dim = ttn.nodes[swap_pair[0]].open_dimension()
                swap_matrix = swap_gate(dimension=dim)
            swap_operator = NumericOperator(swap_matrix, list(swap_pair))
            operator_list.append(swap_operator)
        return operator_list

class TrotterSplitting:
    """
    A trotter splitting allows the approximate breaking of exponentials of operators.
     Different kinds of splitting lead to different error sizes.
    """

    def __init__(self, terms: List[Term],
                 splitting: Union[List[Tuple[int, int], int], None] = None,
                 swaps_before: Union[List[SWAPlist], None] = None,
                 swaps_after: Union[List[SWAPlist], None] = None):
        """Initialises a TrotterSplitting instance.

        Args:
            terms (List[Term]): The terms to be considered.
            splitting (Union[List[Tuple[int, int], int], None], optional): Gives the order
             of the splitting. The first tuple entry is a the index of an operator in
             operators and the second entry is a factor, which will be multiplied to the
             operator once exponentiated. If only an integer is given, it is assumed to be
             the index in the operator list and the factor is set to 1. In case of no given
             splitting the splitting is assumed to be in the order as given in the operator
             list and all factors are set to 1. Defaults to None.
            swaps_before (Union[List[SWAPlist], None], optional): The swaps to be applied
             before an exponentiated operator is applied. The indices are the same as in the
             splitting. So the SWAP gates given with index `i` will be applied before the
             operator specified with the `i`th element of splitting happens. Defaults to None.
            swaps_after (Union[List[SWAPlist], None], optional): The swaps to be applied
             after an exponentiated operator is applied. The indices are the same as in the
             splitting. So the SWAP gates given with index `i` will be applied after the
             operator specified with the `i`th element of splitting happens. Defaults to None.

        Raises:
            TypeError: Raised if the splitting contains unallowed types.
        """
        self.terms = terms

        if splitting is None:
            # Default splitting
            self.splitting = [(index, 1) for index in range(len(terms))]
        else:
            self.splitting = []
            for item in splitting:
                if isinstance(item, int):
                    self.splitting.append((item, 1))
                elif isinstance(item, tuple):
                    self.splitting.append(item)
                else:
                    errstr = "Items in the `splitting` list may only be int or tuple with length 2!"
                    raise TypeError(errstr)

        if swaps_before is None:
            self.swaps_before = [SWAPlist([])] * len(self.splitting)
        else:
            self.swaps_before = swaps_before
        if swaps_after is None:
            self.swaps_after = [SWAPlist([])] * len(self.splitting)
        else:
            self.swaps_after = swaps_after

    def is_compatible_with_ttn(self, ttn):
        """
        Checks, if this splitting is compatible with a given TreeTensorNetwork.
        This means it checks if all sites to which operators should be applied
        are in the TTN and have correct physicial dimension.
        Furthermoe checks if all nodes in the SWAP-list are actually neighbours
        with the same open leg dimension.

        Parameters
        ----------
        ttn : TreeTensorNetwork
            A TTN for which to check compatability.

        Returns
        -------
        compatible: bool
        True if the TrotterSplitting is compatible with the TTN and False,
        if not.

        """
        for interaction_operator in self.operators:
            for site_id in interaction_operator:
                # Check if all operator sites are in the TTN
                if not (site_id in ttn.nodes):
                    return False

                # Check dimensional compatability
                node = ttn.nodes[site_id]
                local_operator = interaction_operator[site_id]
                if node.open_dimension() != local_operator.shape[0]:
                    return False

        # Check compatability of all SWAP lists
        for swap_list in self.swaps_before:
            if not swap_list.is_compatible_with_ttn(ttn):
                return False

        for swap_list in self.swaps_after:
            if not swap_list.is_compatible_with_ttn(ttn):
                return False

        return True

    def exponentiate_splitting(self, ttn: TreeTensorNetwork, delta_time: float,
                               dim: Union[int, None] = None):
        """
        Computes all operators, which are to actually be applied in a time-
        evolution algorithm. This includes SWAP gates and exponentiated
        operators.

        Parameters
        ----------
        ttn : TreeTensorNetwork
            A TTN which is compatible with this splitting.
            Provides the required dimensionality for the SWAPs.
        delta_time : float
            The time step size for the trotter-splitting.
        dim : int, optional
            If all nodes have the same physical dimension, it can be provided
            here. Speeds up the computation especially for big TTN.
            The default is None.

        Returns
        -------
        unitary_operators : list of dict
            All operators that make up one time-step of the Trotter splitting.
            They are to be applied according to their index order in the list.
            Each operator is saved as a dictionary, where the actual operator
            is saved as an ndarray under the key `"operator"` and the sites it
            is applied to are saved as a list of strings/site identifiers under
            they key `"site_ids"`.
            """

        unitary_operators = [] # Includes the neccessary SWAPs
        for i, split in enumerate(self.splitting):
            term = self.terms[split[0]]
            factor = split[1]
            total_operator = 1 # Saves the total operator
            site_ids = [] # Saves the ids of nodes to which the operator is applied
            for node_id, single_site_operator in term.items():
                total_operator = np.kron(total_operator,
                                              single_site_operator)
                site_ids.append(node_id)
            exponentiated_operator = expm((-1j*factor*delta_time) * total_operator)
            exponentiated_operator = NumericOperator(exponentiated_operator, site_ids)

            # Build required swaps for befor trotter term
            swaps_before = self.swaps_before[i].into_operators(ttn=ttn, dim=dim)
            # Build required swaps for after trotter term
            swaps_after = self.swaps_after[i].into_operators(ttn=ttn, dim=dim)

            # Add all operators associated with this term to the list of unitaries
            unitary_operators.extend(swaps_before)
            unitary_operators.append(exponentiated_operator)
            unitary_operators.extend(swaps_after)
        return unitary_operators
