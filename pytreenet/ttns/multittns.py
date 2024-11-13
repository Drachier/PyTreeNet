import numpy as np
from typing import List, Union, Callable
from uuid import uuid1
from copy import deepcopy
from ..ttns.ttns import TreeTensorNetworkState
from ..core.ttn import TreeTensorNetwork
from ..core.leg_specification import LegSpecification
from ..core.canonical_form import _build_qr_leg_specs
from ..core.node import Node
from ..util.tensor_splitting import (tensor_qr_decomposition,
                                     contr_truncated_svd_splitting,
                                     idiots_splitting,
                                     SplitMode,
                                     SVDParameters)
from ..ttno import TTNO
from ..operators.tensorproduct import TensorProduct
from ..contractions.state_state_contraction import contract_two_ttns
from ..contractions.state_operator_contraction import expectation_value
from .utils_multittns import (_init_orthogonal_states, _scalar_product_multittn, 
                           )

class MultiTreeTensorNetworkState(TreeTensorNetworkState):
    """
    A class representing a multi-state tree tensor network state.
    
    A multi-state tree tensor network state is a tree tensor network state
    with multiple states at one node. And the multiple state node can be  swept
    through the network. The multi-state node is also the orthogonality center.
    """
    def __init__(self, weight: Union[List[float], List[int], np.ndarray])-> None:
        super().__init__()
        if isinstance(weight, list):
            weight = np.array(weight)
        self.weight = weight
        self.num_states = len(weight)
        
    def init_multistate_center(self, list_tensor: List[np.ndarray]=None)-> None:
        """
        Initialize a node with multiple states.
        """
        if self.orthogonality_center_id is None:
            raise ValueError("Orthogonality center does not exist. Please set the orthogonality center first.")
        identifier = self.orthogonality_center_id
        self.state_tensors = _init_orthogonal_states(self.tensors[identifier], self.num_states, list_tensor)
        self.state_center_id = identifier
        self.tensors[identifier] = self.state_tensors[0]
    
    def scalar_product(self, other: Union[TreeTensorNetworkState, None]=None,
                       use_orthogonality_center: bool=True)-> complex:
        """
        Compute the scalar product of the multi-state tree tensor network state.
        """
        if other is None:
            if self.state_center_id == self.orthogonality_center_id and use_orthogonality_center:
                return _scalar_product_multittn(self.state_tensors, self.weight)
            other = deepcopy(self)
        # Very inefficient, fix later without copy
        return _contract_multittns_ttns(self, other.conjugate())
    
    def single_site_operator_expectation_value(self, node_id: str,
                                               operator: np.ndarray) -> List[complex]:
        """
        The expectation value with regards to a single-site operator.

        The single-site operator acts on the specified node.

        Args:
            node_id (str): The identifier of the node, the operator is applied
                to.
            operator (np.ndarray): The operator of which we determine the
                expectation value. Note that the state will be contracted with
                axis/leg 1 of this operator.

        Returns:
            complex: The resulting expectation value < TTNS| Operator| TTN >.
        """
        if self.orthogonality_center_id == node_id and self.state_center_id == node_id:
            expectation_value = []
            for i in range(self.num_states):
                tensor = deepcopy(self.state_tensors[i])
                tensor_op = np.tensordot(tensor, operator, axes=(-1,1))
                tensor_conj = tensor.conj()
                legs = tuple(range(tensor.ndim))
                expectation_value.append(self.weight[i]*complex(np.tensordot(tensor_op, tensor_conj, axes=(legs,legs))))
            return expectation_value
        tensor_product = TensorProduct({node_id: operator})
        return self.operator_expectation_value(tensor_product)
    
    def operator_expectation_value(self, operator: Union[TensorProduct,TTNO]) -> List[complex]:
        """
        Finds the expectation value of the operator specified, given this TTNS.

        Args:
            operator (Union[TensorProduct,TTNO]): A TensorProduct representing
                the operator as many single site operators. Otherwise a a TTNO
                with the same structure as the TTNS.

        Returns:
            complex: The resulting expectation value < TTNS | operator | TTNS>
        """
        if isinstance(operator, TensorProduct):
            if len(operator) == 0:
                return self.scalar_product()
            if len(operator) == 1:
                node_id = list(operator.keys())[0]
                if self.orthogonality_center_id == node_id:
                    op = operator[node_id]
                    return self.single_site_operator_expectation_value(node_id, op)
            # Very inefficient, fix later without copy
            ttn = deepcopy(self)         
            expectation_value = []
            for i in range(self.num_states):
                ttn.tensors[ttn.state_center_id] = ttn.state_tensors[i]
                conj_ttn = ttn.conjugate()
                for node_id, single_site_operator in operator.items():
                    ttn.absorb_into_open_legs(node_id, single_site_operator)
                expectation_value.append(contract_two_ttns(ttn, conj_ttn))
            return expectation_value
        # Operator is a TTNO
        return _expectation_value_multittn(self, operator)

    def apply_operator(self, operator: TensorProduct):
        """
        Applies a tensor product operator to the TTNS.

        Args:
            operator (TensorProduct): The operator to apply.
        """
        for node_id, single_site_operator in operator.items():
            _absorb_into_open_legs_multittn(self, node_id, single_site_operator)
    
    def split_nodes_multittns(self, node_id: str,
                     out_legs: LegSpecification, in_legs: LegSpecification,
                     splitting_function: Callable,
                     out_identifier: str = "", in_identifier: str = "",
                     **kwargs):
        """
        Splits orthogonality center node into two nodes using a specified function

        Args:
            node_id (str): The identifier of the node to be split.
            out_legs (LegSpecification): The legs associated to the output of the
                matricised node tensor. (The Q legs for QR and U legs for SVD)
            in_legs (LegSpecification): The legs associated to the input of the
                matricised node tensor: (The R legs for QR and the SVh legs for SVD)
            splitting_function (Callable): The function to be used for the splitting
                of the tensor. This function should take the tensor and the
                legs in the form of integers and return two tensors. The first
                tensor should have the legs in the order
                (parent_leg, children_legs, open_legs, new_leg) and the second
                tensor should have the legs in the order
                (new_leg, parent_leg, children_legs, open_legs).
            out_identifier (str, optional): An identifier for the tensor with the
            output legs. Defaults to "".
            in_identifier (str, optional): An identifier for the tensor with the input
                legs. Defaults to "".
            **kwargs: Are passed to the splitting function.
        """
        if self.state_center_id != node_id:
            raise ValueError("Only the orthogonality center node can be split")
        node, tensor = self[node_id]
        assert tensor.shape == self.state_tensors[0].shape, f"Tensor shape mismatch: {tensor.shape} != {self.state_tensors[0].shape}"
        tensor_list = [deepcopy(self.state_tensors[i]) for i in range(self.num_states)]
        state_tensors = np.stack(tensor_list,axis=-1)
        
        if out_legs.node is None:
            out_legs.node = node
        if in_legs.node is None:
            in_legs.node = node
        # Find new identifiers
        if out_identifier == "":
            out_identifier = "out_of_" + node_id
        if in_identifier == "":
            in_identifier = "in_of_" + node_id

        # Getting the numerical value of the legs
        out_legs_int = out_legs.find_leg_values()
        in_legs_int = in_legs.find_leg_values()
        in_legs_int.append(len(tensor.shape))
        out_tensor, in_tensors = splitting_function(state_tensors,
                                                   out_legs_int,
                                                   in_legs_int,
                                                   **kwargs)
        self._tensors[out_identifier] = out_tensor
        in_tensor = in_tensors[:,:,0]
        
        self._tensors[in_identifier] = in_tensor

        # New Nodes
        out_node = Node(tensor=out_tensor, identifier=out_identifier)
        in_node = Node(tensor=in_tensor, identifier=in_identifier)
        self._nodes[out_identifier] = out_node
        self._nodes[in_identifier] = in_node
        self.state_tensors_intermediate = in_tensors

        # Currently the tensors out and in have the leg ordering
        # (new_leg(for in), parent_leg, children_legs, open_legs, new_leg(for out))
        self._set_in_parent_leg_after_split(in_node,
                                            in_legs,
                                            out_identifier)
        self._set_in_children_legs_after_split(in_legs,
                                               out_legs,
                                               in_identifier,
                                               out_identifier)
        self._set_out_parent_leg_after_split(out_node,
                                             out_legs,
                                             in_identifier)
        self._set_out_children_legs_after_split(out_legs,
                                                in_legs,
                                                out_identifier,
                                                in_identifier)
        self.replace_node_in_some_neighbours(out_identifier, node_id,
                                             out_legs.find_all_neighbour_ids())
        self.replace_node_in_some_neighbours(in_identifier, node_id,
                                             in_legs.find_all_neighbour_ids())
        self._set_root_from_leg_specs(in_legs, out_legs,
                                      in_identifier, out_identifier)
        if node_id not in [out_identifier, in_identifier]:
            self._tensors.pop(node_id)
            self._nodes.pop(node_id)
        
        perm_order = in_node.leg_permutation  
        self.state_tensors_intermediate = [np.einsum(in_tensors[:,:,i], perm_order) for i in range(self.num_states)]
        self.state_tensors_intermediate = np.stack(self.state_tensors_intermediate, axis=-1)
    
    def move_orthogonalization_center_multittns(self, new_center_id: str,
                                      split_method: str = "qr",
                                      mode: SplitMode = SplitMode.REDUCED,
                                      svd_params: SVDParameters = SVDParameters()):
        """
        Moves the orthogonalization center to a different node.

        For this to work the TTN has to be in a canonical form already, i.e.,
        there should already be an orthogonalisation center.

        Args:
            new_center (str): The identifier of the new orthogonalisation
                center.
            mode: The mode to be used for the QR decomposition. For details refer to
                `tensor_util.tensor_qr_decomposition`.
        """
        if self.orthogonality_center_id is None:
            errstr = "The TTN is not in canonical form, so the orth. center cannot be moved!"
            raise AssertionError(errstr)
        if self.orthogonality_center_id == new_center_id:
            # In this case we are done already.
            return
        orth_path = self.path_from_to(self.orthogonality_center_id,
                                      new_center_id)
        for node_id in orth_path[1:]:
            self._move_orth_center_to_neighbour_multittns(node_id, split_method=split_method, mode=mode, svd_params=svd_params)    
        
    def _move_orth_center_to_neighbour_multittns(self, new_center_id: str, split_method: str = "qr",
                                       mode: SplitMode = SplitMode.REDUCED, 
                                       svd_params: SVDParameters = SVDParameters()):
        """
        Moves the orthogonality center to a neighbour of the current center.

        Args:
            new_center_id (str): The identifier of a neighbour of the current
                orthogonality center.
            split_method (str): The method to be used for the splitting. It can be
                "qr" or "svd".
            mode: The mode to be used for the QR decomposition. For details refer to
                `tensor_util.tensor_qr_decomposition`.
        """
        assert self.orthogonality_center_id is not None
        node = self.nodes[self.orthogonality_center_id]
        q_legs, r_legs = _build_qr_leg_specs(node, new_center_id)
        r_tensor_id = str(uuid1())
        if split_method == "qr":
            self.split_nodes_multittns(self.orthogonality_center_id, q_legs, r_legs, 
                                       tensor_qr_decomposition,
                                       out_identifier=self.orthogonality_center_id,
                                       in_identifier=r_tensor_id,
                                        mode=mode)
        elif split_method == "svd":
            self.split_nodes_multittns(self.orthogonality_center_id, q_legs, r_legs, 
                                       contr_truncated_svd_splitting, 
                                       out_identifier=self.orthogonality_center_id,
                                       in_identifier=r_tensor_id,
                                       svd_params=svd_params)

        # todo: contract with all states
        new_state_tensors = []
        assert np.allclose(self.tensors[r_tensor_id], self.state_tensors_intermediate[:,:,0])
        for i in range(self.num_states):
            self.tensors[r_tensor_id] = self.state_tensors_intermediate[:,:,i]
            new_tensor = self._contract_without_deleting_nodes(new_center_id, r_tensor_id)
            new_state_tensors.append(new_tensor)
        # self.state_tensors_intermediate = np.stack(new_state_tensors, axis=-1)
        self.tensors[r_tensor_id] = self.state_tensors_intermediate[:,:,0]
        self.contract_nodes(new_center_id, r_tensor_id,new_identifier=new_center_id)
        # assert np.allclose(self.tensors[new_center_id], new_state_tensors[0])
        leg_permutation = self.nodes[new_center_id].leg_permutation
        self.state_tensors = [np.einsum(tensor, leg_permutation) for tensor in new_state_tensors]
        # self.state_tensors = deepcopy(new_state_tensors)  

        self.orthogonality_center_id = new_center_id
        self.state_center_id = new_center_id
        
    def _contract_without_deleting_nodes(self, node_id1: str, node_id2: str)-> np.ndarray:
        """
        Contract two nodes without deleting any nodes.
        """
        parent_id, child_id = self.determine_parentage(node_id1, node_id2)
        parent_node = self.nodes[parent_id]
        parent_tensor = self.tensors[parent_id]
        child_tensor = self.tensors[child_id]
        axes = (parent_node.neighbour_index(child_id), 0)
        new_tensor = np.tensordot(parent_tensor, child_tensor, # This order for leg convention
                                  axes=axes)
        return new_tensor
        
MultiTTNS = MultiTreeTensorNetworkState

def _expectation_value_multittn(ttn: MultiTreeTensorNetworkState, operator: TTNO)-> List[complex]:
    """
    Compute the expectation value of a multi-state tree tensor network state.
    """
    ttn_copy = deepcopy(ttn)
    assert ttn_copy.state_center_id == ttn_copy.orthogonality_center_id, "Orthogonality center and state center must be the same"
    expectvals = []
    for i in range (ttn.num_states):
        ttn_copy.tensors[ttn_copy.state_center_id] = ttn_copy.state_tensors[i]
        expectvals.append(expectation_value(ttn_copy, operator))
    return expectvals

def _contract_multittns_ttns(ttn1: MultiTreeTensorNetworkState, ttn2: TreeTensorNetworkState)-> List[complex]:
    """
    Contract a multi-state tree tensor network state with a tree tensor network state.
    """
    ttn1_copy = deepcopy(ttn1)
    assert ttn1_copy.state_center_id == ttn1_copy.orthogonality_center_id, "Orthogonality center and state center must be the same"
    contract_values = []
    for i in range(ttn1_copy.num_states):
        ttn1_copy.tensors[ttn1_copy.state_center_id] = ttn1_copy.state_tensors[i]
        contract_values.append(contract_two_ttns(ttn1_copy, ttn2))
    return contract_values

def _absorb_into_open_legs_multittn(ttn: MultiTreeTensorNetworkState, node_id: str, tensor: np.ndarray)-> None:
    """
    Absorb a tensor into the open legs of the tensor of a node.

    This tensor will be absorbed into all open legs and it is assumed, the
    leg order of the tensor to be absorbed is the same as the order of
    the open legs of the node.

    Since the tensor to be absorbed is considered to represent an operator
    acting on the node, it will have to have exactly twice as many legs as
    the node has open legs. The input legs, i.e. the ones contracted, are
    assumed to be the second half of the legs of the tensor.

    Args:
        node_id (str): The identifier of the node which is to be contracted
            with the tensor
        tensor (np.ndarray): The tensor to be contracted.
    """
    node, node_tensor = ttn[node_id]
    nopen_legs = node.nopen_legs()
    assert tensor.ndim == 2 * nopen_legs
    if tensor.shape[:nopen_legs] != tensor.shape[nopen_legs:]:
        errstr = self._absorption_warning()
        raise AssertionError(errstr)
    tensor_legs = [i + nopen_legs for i in range(nopen_legs)]
    if ttn.state_center_id != node_id:
        new_tensor = np.tensordot(node_tensor, tensor,
                                axes=(node.open_legs, tensor_legs))
        # The leg ordering was not changed here
        ttn.tensors[node_id] = new_tensor
    else:
        for i in range(ttn.num_states):
            new_tensor = np.tensordot(ttn.state_tensors[i], tensor,
                                axes=(node.open_legs, tensor_legs))
            ttn.state_tensors[i] = new_tensor
        ttn.tensors[node_id] = ttn.state_tensors[0]

        