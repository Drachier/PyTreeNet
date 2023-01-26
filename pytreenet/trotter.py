import numpy as np

from .util import build_swap_gate
from scipy.linalg import expm

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

        list.__init__(swap_list)

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
            if not (swap_pair[0] in ttn.nodes):
                return False

            # If it is check, if the other is actually connected and thus also
            #  in the TTN
            node1 = ttn.nodes[swap_pair[0]]
            if not (swap_pair[1] in node1.neighbouring_nodes(with_leg=False)):
                return False

            # Finally check if both have the same total physical dimension.
            node2 = ttn.nodes[swap_pair[1]]
            if node1.open_dimension() != node2.open_dimension():
                return False

        return True

class TrotterSplitting:
    def __init__(self, operators, splitting=None, swaps_before=None, swaps_after=None):
        """

        Parameters
        ----------
        operators : list of dict
            A list of dictionaries, where each dictionary represents an operator
            with finite support. The values are local single_site operators and
            the corresponding keys are the identifiers of the sites to which
            they are to be applied.
        splitting : list of tuples and tuple, optional
            A list of tuples which considers the order of the Trotter splitting.
            The first entry of each tuple is the index of an operator in the
            operators list. The second entry is a factor.
            The order in which the tuples appear is the order in which the
            operators appear in the splitting, while the factor is a factor
            multiplied to the time in the exponential. If 'splitting' is a list
            of int every factor is assumed to be 1. If no splitting is provided
            the order is taken as the order in operators.
            Defualt is None.
        swaps_before, swaps_after : list of SWAPlist, optional
            A list of all the SWAPSlists that are to happen before/after each Trotter
            step. The index of the SWAPlist corresponds to the index of the
            tuple in 'splitting' before/after which it is to be applied.
            The default is None.
        """

        self.operators = operators

        if splitting == None:
            self.splitting = [(index, 1) for index in range(len(operators))]
        else:
            self.splitting = []
            for item in splitting:
                if type(item) == int:
                    self.splitting.append((item, 1))
                elif type(item) == tuple:
                    if len(item) == 1:
                        self.splitting.append((item[0],1))
                    elif len(item) == 2:
                        self.splitting.append(item)
                    else:
                        raise TypeError("Items in the `splitting` list may only be int or tuple with len 2")
                else:
                    raise TypeError("Items in the `splitting` list may only be int or tuple with len 2")


        if swaps_before == None:
            self.swaps_before = [SWAPlist([]) for i in splitting]
        else:
            self.swaps_before = swaps_before

        if swaps_after == None:
            self.swaps_after = [SWAPlist([]) for i in splitting]
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

    def exponentiate_splitting(self, ttn, delta_time, dim=None):
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
            herer. Speeds up the computation especially for big TTN.
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

        for i, term in enumerate(self.splitting):

            interaction_operator = self.operators[term[0]]
            factor = term[1]

            total_operator = 1 # Saves the total operator
            site_ids = [] # Saves the ids of nodes to which the operator is applied

            for site in interaction_operator:
                total_operator = np.kron(total_operator,
                                              interaction_operator[site])

                site_ids.append(site)

            exponentiated_operator = expm((-1j*factor*delta_time) * total_operator)
            exponentiated_operator = {"operator": exponentiated_operator,
                                      "site_ids": site_ids}

            # Build required swaps for befor trotter term
            swaps_before = []

            for swap_pair in self.swaps_before[i]:
                if dim == None:
                    dimension = ttn.nodes[swap_pair[0]].open_dimension()
                else:
                    dimension = dim

                swap_gate = build_swap_gate(dimension=dimension)

                swap_operator = {"operator": swap_gate,
                                 "site_ids": list(swap_pair)}

                swaps_before.append(swap_operator)

            # Build required swaps for after trotter term
            swaps_after = []

            for swap_pair in self.swaps_after[i]:
                if dim == None:
                    dimension = ttn.nodes[swap_pair[0]].open_dimension()
                else:
                    dimension = dim

                swap_gate = build_swap_gate(dimension=dimension)

                swap_operator = {"operator": swap_gate,
                                 "site_ids": list(swap_pair)}

                swaps_after.append(swap_operator)

            # Add all operators associated with this term to the list of unitaries
            unitary_operators.extend(swaps_before)
            unitary_operators.append(exponentiated_operator)
            unitary_operators.extend(swaps_after)

        return unitary_operators

